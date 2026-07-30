"""Find flight sessions without being told where they are.

Every file the logger writes carries a ``YYYYMMDD_HHMMSS`` stamp, so the data
directory is already an index -- it just is not queryable. This module reads
that structure back out: it walks the tree, pairs each flight log with the
camera run recorded alongside it, and hands back `Session` objects you select
by name instead of by path.

    catalog.resolve("latest")       newest session
    catalog.resolve("0723.last")    last run of 23 Jul
    catalog.resolve("114556")       by timestamp fragment

Nothing here is written to disk and nothing needs maintaining: the walk costs
about half a second over the whole archive, so the index is rebuilt on every
call rather than cached and left to go stale. The one thing that cannot be
derived from a filename -- the pose/flight clock offset, the tether lengths,
your notes -- lives in sessions.toml next to this file.
"""
import datetime as _dt
import os
import re
import tomllib
from functools import cached_property
from pathlib import Path

# Root of the archive. Point TARES_DATA elsewhere to work off a copy.
DATA_ROOT = Path(os.environ.get("TARES_DATA", "~/TARES_SITL/data")).expanduser()

# Hand-written facts that no filename knows.
META_FILE = Path(__file__).with_name("sessions.toml")

# A camera run and its flight log start within a second or so of each other.
PAIR_TOLERANCE_S = 5.0

_TS = re.compile(r"(\d{8})_(\d{6})")

# Columns worth summarising, by their current names. The logger was renamed
# at some point -- the June 2026 logs call these t/mode/pz/px_ref -- so the
# older spelling maps onto the newer one rather than reading as an empty file.
_SUMMARY_COLS = ("cur_time", "echoed_mode", "drone_pz_meas",
                 "drone_px_ref", "payload_px_ref")
_LEGACY_NAMES = {"t": "cur_time",
                 "mode": "echoed_mode",
                 "pz": "drone_pz_meas",
                 "px_ref": "drone_px_ref"}


def _metadata():
    """The whole sessions.toml document, or an empty one."""
    if not META_FILE.exists():
        return {}
    with open(META_FILE, "rb") as f:
        return tomllib.load(f)


def defaults():
    """The [defaults] table: what holds for every session unless overridden."""
    return _metadata().get("defaults", {})


def _timestamp(name):
    """Parse YYYYMMDD_HHMMSS out of a file or directory name."""
    m = _TS.search(name)
    if not m:
        return None
    return _dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")


class Session:
    """One flight: its log, the camera run beside it, and what we know."""

    def __init__(self, sid, time, flight, pose=None, duplicates=()):
        self.id = sid                       # '20260723_114556'
        self.time = time
        self.flight = flight
        self.pose = pose
        self.duplicates = list(duplicates)  # same log filed in another folder

    # -- identity ----------------------------------------------------------
    @property
    def date(self):
        return self.time.strftime("%m%d")

    @property
    def has_camera(self):
        return self.pose is not None

    @property
    def is_sim(self):
        """ArduPilot SITL rather than a real flight. Only sessions.toml knows."""
        return bool(self.meta.get("sim", False))

    @property
    def flags(self):
        return ",".join(f for f, on in (("cam", self.has_camera),
                                        ("sim", self.is_sim)) if on) or "-"

    @property
    def events(self):
        p = self.flight.with_name(self.flight.stem + "_events.csv")
        return p if p.exists() else None

    @property
    def label(self):
        """Human name: the camera run folder if there is one, else the log."""
        return self.pose.parent.name if self.has_camera else self.flight.stem

    def __repr__(self):
        return f"<Session {self.id}{' +cam' if self.has_camera else ''}>"

    # -- the facts we cannot derive ----------------------------------------
    @cached_property
    def meta(self):
        """sessions.toml defaults, overridden by this session's own entry."""
        doc = _metadata()
        return {**doc.get("defaults", {}),
                **doc.get("sessions", {}).get(self.id, {})}

    def __getattr__(self, name):
        # session.pose_offset, session.L_marker, ... come from sessions.toml
        try:
            return self.meta[name]
        except KeyError:
            raise AttributeError(
                f"{name!r} is not a column of this Session and is not set for "
                f"{self.id} in {META_FILE.name}") from None

    # -- the data ----------------------------------------------------------
    @cached_property
    def fl(self):
        """The flight log as a DataFrame."""
        import pandas as pd
        return pd.read_csv(self.flight)

    @cached_property
    def poses(self):
        """The camera pose CSV as a DataFrame."""
        if not self.has_camera:
            raise FileNotFoundError(f"session {self.id} has no camera data")
        import pandas as pd
        return pd.read_csv(self.pose)

    @cached_property
    def summary(self):
        """Cheap facts read off the log: duration, modes, altitude, target."""
        import pandas as pd
        out = {"rows": 0, "duration": float("nan"), "modes": "?",
               "alt": float("nan"), "ref": "?",
               "mb": self.flight.stat().st_size / 1e6}
        wanted = set(_SUMMARY_COLS) | set(_LEGACY_NAMES)
        try:
            d = pd.read_csv(self.flight, usecols=lambda c: c in wanted)
        except Exception:
            out["modes"] = "unreadable"
            return out
        d = d.rename(columns={old: new for old, new in _LEGACY_NAMES.items()
                              if old in d and new not in d})
        if "cur_time" not in d or not len(d):
            out["modes"] = "empty"
            return out
        out["rows"] = len(d)
        out["duration"] = float(d.cur_time.iloc[-1])
        if "echoed_mode" in d:
            modes = [m for m in pd.unique(d.echoed_mode.astype(str))
                     if m not in ("?", "nan", "None")]
            out["modes"] = "/".join(modes) or "?"
        if "drone_pz_meas" in d:
            out["alt"] = float(d.drone_pz_meas.max())
        # the log carries both reference sets; the one that was flown is filled
        out["ref"] = _flown_reference(d)
        return out


def _flown_reference(d):
    """Which reference the flight actually tracked, read off the log."""
    def used(col):
        return col in d and d[col].notna().any() and d[col].abs().max() > 0
    if used("payload_px_ref"):
        return "payload"
    if used("drone_px_ref"):
        return "drone"
    return "?"


# ----------------------------------------------------------------------------
# discovery
# ----------------------------------------------------------------------------
def _flight_logs(root):
    """{timestamp id: [paths]} for every flight log under root."""
    found = {}
    for p in root.rglob("flight_*.csv"):
        if p.name.endswith("_events.csv"):
            continue
        t = _timestamp(p.name)
        if t is not None:
            found.setdefault(t.strftime("%Y%m%d_%H%M%S"), []).append(p)
    return found


def _pose_files(root):
    """[(timestamp, path)] for camera runs that carry a stamp."""
    out = []
    for p in root.rglob("poses*.csv"):
        t = _timestamp(p.parent.name) or _timestamp(p.name)
        if t is not None:
            out.append((t, p))
    return out


def sessions(root=None):
    """Every flight session under `root`, oldest first."""
    root = Path(root).expanduser() if root else DATA_ROOT
    if not root.is_dir():
        raise FileNotFoundError(f"data root not found: {root}")

    logs = _flight_logs(root)
    poses = _pose_files(root)

    out = []
    for sid, paths in logs.items():
        # the same log often sits in two folders; prefer the filed copy, the
        # one nested deepest, and keep the strays so `ls` can flag them
        paths.sort(key=lambda p: (-len(p.parts), str(p)))
        t = _dt.datetime.strptime(sid, "%Y%m%d_%H%M%S")

        pose, best = None, PAIR_TOLERANCE_S
        for pt, pp in poses:
            dt = abs((pt - t).total_seconds())
            if dt <= best:
                pose, best = pp, dt

        out.append(Session(sid, t, paths[0], pose, paths[1:]))

    out.sort(key=lambda s: s.time)
    return out


def unpaired_poses(root=None):
    """Camera CSVs with no timestamp, so nothing to attach them to."""
    root = Path(root).expanduser() if root else DATA_ROOT
    return sorted(p for p in root.rglob("poses*.csv")
                  if _timestamp(p.parent.name) is None
                  and _timestamp(p.name) is None)


# ----------------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------------
_FILTERS = {"cam": lambda s: s.has_camera,
            "nocam": lambda s: not s.has_camera,
            "sim": lambda s: s.is_sim,
            "real": lambda s: not s.is_sim}


def select(selector, pool=None):
    """Sessions matching `selector`, oldest first.

    The selector is dot-separated: a base set, then filters and ordinals.

        latest              the newest session
        all                 everything
        0723                every session on 23 Jul (or 20260723)
        0723.cam            ...that has camera data
        0723.cam.last       ...the last of those
        0723.2              the third (0-based, as printed by `ls`)
        114556              anything whose stamp contains 114556
    """
    pool = list(pool) if pool is not None else sessions()
    if not selector:
        return pool

    for token in str(selector).split("."):
        if not pool:
            break
        if token in ("all", ""):
            continue
        elif token == "latest":
            pool = pool[-1:]
        elif token == "last":
            pool = pool[-1:]
        elif token == "first":
            pool = pool[:1]
        elif token in _FILTERS:
            pool = [s for s in pool if _FILTERS[token](s)]
        elif token.isdigit() and len(token) <= 3:
            i = int(token)
            pool = [pool[i]] if i < len(pool) else []
        elif token.isdigit():
            pool = [s for s in pool if token in s.id]
        else:
            low = token.lower()
            pool = [s for s in pool
                    if low in s.label.lower() or low in str(s.flight).lower()]
    return pool


def resolve(selector, pool=None):
    """Exactly one session, or an error that says what else matched."""
    hits = select(selector, pool)
    if not hits:
        raise LookupError(f"no session matches {selector!r} under {DATA_ROOT}"
                          f"\ntry:  uv run plot.py ls")
    if len(hits) > 1:
        names = "\n  ".join(f"{s.id}  {s.label}" for s in hits[:12])
        more = "" if len(hits) <= 12 else f"\n  ... and {len(hits)-12} more"
        raise LookupError(
            f"{selector!r} matches {len(hits)} sessions; add .last, .first or "
            f"an index:\n  {names}{more}")
    return hits[0]


# ----------------------------------------------------------------------------
def format_table(pool):
    """The `ls` listing: one line per session, newest last."""
    head = (f"{'#':>3}  {'session':16}  {'dur':>7}  {'rows':>6}  {'alt':>6}  "
            f"{'ref':>7}  {'mode':12}  {'flags':7}  where")
    lines = [head, "-" * len(head)]
    for i, s in enumerate(pool):
        d = s.summary
        where = str(s.flight.parent.relative_to(DATA_ROOT))
        dup = f"  [+{len(s.duplicates)} dup]" if s.duplicates else ""
        lines.append(
            f"{i:>3}  {s.id:16}  {d['duration']:6.1f}s  {d['rows']:>6}  "
            f"{d['alt']:6.1f}  {d['ref']:>7}  {d['modes'][:12]:12}  "
            f"{s.flags:7}  {where}{dup}")
    return "\n".join(lines)
