"""
Find flight sessions without being told where they are
"""
import datetime as _dt
import os
import re
import tomllib
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

# Root of the archive. Point TARES_DATA elsewhere to work off a copy.
DATA_ROOT = Path(os.environ.get("TARES_DATA", "~/TARES/data")).expanduser()

# Hand-written facts that no filename knows.
META_FILE = Path(__file__).with_name("sessions.toml")

# camera CSVs, one name per tracker: markers write poses, color writes circles
TRACK_GLOBS = ("poses*.csv", "circles*.csv")

# Pairing a camera run with its flight log, by folder stamp. The stamp is
# taken when the recorder starts, which is before the camera device is open,
# so it can sit well ahead of the first frame -- only good to a few seconds.
PAIR_TOLERANCE_S = 5

# Pairing on the wall_time both files now carry. That is the same clock read
# in both, so the slack is only for the order the two were started in.
WALL_TOLERANCE_S = 60

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
    """
    The whole sessions.toml document
    """
    if not META_FILE.exists():
        return {}

    with open(META_FILE, "rb") as f:
        return tomllib.load(f)


def defaults():
    """
    The [defaults] table
    """
    return _metadata().get("defaults", {})


def _as_date(token):
    """
    A YYYYMMDD key from MMDDYYYY  or YYYYMMDD.
    """
    for fmt in ("%m%d%Y", "%Y%m%d"):
        try:
            return _dt.datetime.strptime(token, fmt).strftime("%Y%m%d")

        except ValueError:
            continue

    return None


def _first(path, column):
    """
    The first row's column
    """
    try:
        d = pd.read_csv(path, usecols=[column], nrows=1)

    except (ValueError, OSError):
        return None

    return float(d[column].iloc[0]) if len(d) else None


def _repair_cur_time(d):
    """
    Rebuild cur_time from wall_time where the logged clock is not monotonic.

    Old flight plans reset cur_time to 0 at the start of every leg
    (fixed in src/comms/control.py). wall_time is stamped alongside it and
    stays continuous regardless, so it stands in when that happened.
    """
    if "cur_time" not in d or "wall_time" not in d or not len(d):
        return d
    ct = d["cur_time"].to_numpy(float)
    if np.all(np.diff(ct) >= -1e-6):
        return d
    d = d.copy()
    d["cur_time"] = d["wall_time"] - d["wall_time"].iloc[0]
    return d


def _clock_epoch(path, elapsed_col):
    """When a log's own clock started: wall_time minus its elapsed column.

    Constant to well under a millisecond down the file -- both come off one
    time.time() call per row -- so the first row settles it.
    """
    wall = _first(path, "wall_time")
    elapsed = _first(path, elapsed_col)
    if wall is None or elapsed is None:
        return None
    return wall - elapsed


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
        """MMDDYYYY, the form the selectors and the data folders use."""
        return self.time.strftime("%m%d%Y")

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

    @cached_property
    def config(self):
        """
        The exact Prm/config.py values this flight ran with.

        Written by FlightLogger next to the flight log. Prm/config.py
        changes between experiments, so raise rather than fall back to it
        for a session recorded before this snapshot existed.
        """
        import json
        path = self.flight.with_name("config_snapshot.json")
        if not path.exists():
            raise FileNotFoundError(
                f"session {self.id} has no config_snapshot.json")
        with open(path) as f:
            snap = json.load(f)
        return snap

    @cached_property
    def pose_offset(self):
        """
        Pose clock -> flight clock [s]: t_flight = t_pose - pose_offset.

        Both logs stamp every row with the same wall clock, so this is just
        the difference of the two epochs. Raises for a session with no
        camera, or that predates wall_time in poses.csv.
        """
        if not self.has_camera:
            raise AttributeError(f"session {self.id} has no camera data")
        e_pose = _clock_epoch(self.pose, "time_s")
        e_flight = _clock_epoch(self.flight, "cur_time")
        if e_pose is None or e_flight is None:
            raise ValueError(
                f"session {self.id} predates wall_time in poses.csv")
        offset = e_flight - e_pose
        return offset

    def __getattr__(self, name):
        # session.note, session.sim, ... come from sessions.toml
        try:
            return self.meta[name]
        except KeyError:
            raise AttributeError(
                f"{name!r} is not a column of this Session and is not set for "
                f"{self.id} in {META_FILE.name}") from None

    # -- the data ----------------------------------------------------------
    @cached_property
    def fl(self):
        """The flight log as a DataFrame, with cur_time repaired if needed."""
        import pandas as pd
        fl = _repair_cur_time(pd.read_csv(self.flight))
        return fl

    @property
    def tracker(self):
        """
        Which tracker wrote this session's camera CSV, aruco or color.

        A marker run stores a pose per marker, a color run stores one ring
        center per frame, so anything that reads marker columns has to check.
        """
        if not self.has_camera:
            return None

        return "color" if self.pose.name.startswith("circles") else "aruco"

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
        wanted = set(_SUMMARY_COLS) | set(_LEGACY_NAMES) | {"wall_time"}
        try:
            d = pd.read_csv(self.flight, usecols=lambda c: c in wanted)
        except Exception:
            out["modes"] = "unreadable"
            return out
        d = d.rename(columns={old: new for old, new in _LEGACY_NAMES.items()
                              if old in d and new not in d})
        d = _repair_cur_time(d)
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
        out["ref"] = flown_reference(d)
        return out


LOGGED_STATE_COLS = ("payload_alpha_x", "payload_alpha_y",
                     "payload_alphadot_x", "payload_alphadot_y")

# added after LOGGED_STATE_COLS, so checked separately
LOGGED_COV_COLS = ("payload_psi_p", "payload_cov_axx", "payload_cov_ayy",
                   "payload_cov_axy", "payload_cov_psipsi")


def has_logged_states(d):
    """Whether the flight computer ran the payload EKF and logged its output.

    Runs from Aug 2026 on fly the filter in the loop and write the swing state
    it was steering on into every row; earlier logs carry the columns but
    leave them at zero, because nothing was filling them. Read off the data
    rather than configured, the same way `flown_reference` is.
    """
    if not set(LOGGED_STATE_COLS) <= set(d.columns):
        return False
    vals = d[list(LOGGED_STATE_COLS)].abs().to_numpy()
    has_states = bool(np.isfinite(vals).any() and np.nanmax(vals) > 0)
    return has_states


def has_logged_covariance(d):
    """Whether this log also carries the onboard filter's psi_p and covariance.

    Numeric as well as present: an Aug 2026 stick run wrote the whole 5x5
    covariance into the four scalar columns, so those rows hold a printed
    array rather than a number. Such a log reads as having no covariance,
    which costs the uncertainty band and nothing else.
    """
    if not set(LOGGED_COV_COLS) <= set(d.columns):
        return False

    cols = d[list(LOGGED_COV_COLS)]
    if not all(pd.api.types.is_numeric_dtype(cols[c]) for c in LOGGED_COV_COLS):
        return False

    has_cov = bool(cols.notna().to_numpy().any())

    return has_cov


def flown_reference(d):
    """Which reference the flight actually tracked: 'payload', 'drone' or '?'.

    Every log carries both sets of columns and the controller fills in only
    the one it was following, so this is read off the data rather than
    configured. A payload reference outranks a drone reference: a run that
    has one was steering the drone in order to put the payload somewhere,
    which is the thing that run was about.
    """
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


def _track_files(root):
    """Every camera CSV under root, whichever tracker wrote it."""
    return [p for pattern in TRACK_GLOBS for p in root.rglob(pattern)]


def _pose_files(root):
    """[(timestamp, path)] for camera runs that carry a stamp."""
    out = []
    for p in _track_files(root):
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

    _pair_by_wall_clock(out, poses)
    out.sort(key=lambda s: s.time)
    return out


def _pair_by_wall_clock(pool, poses):
    """Attach the camera runs whose folder stamp landed too far off to pair.

    The recorder stamps its folder when it starts and writes its first row
    once the camera is actually open, which can be a quarter minute later --
    far enough that the stamps of a run and its flight log disagree by more
    than PAIR_TOLERANCE_S. The two files share a wall clock, so whatever the
    first pass left over is matched on that instead. Runs older than the
    wall_time column have nothing to match on and stay unpaired, exactly as
    before.
    """
    attached = {s.pose for s in pool if s.has_camera}
    left = [(p, _first(p, "wall_time")) for _, p in poses if p not in attached]
    left = [(p, w) for p, w in left if w is not None]
    if not left:
        return

    free = [(s, _first(s.flight, "wall_time"))
            for s in pool if not s.has_camera]
    free = [(s, w) for s, w in free if w is not None]

    for p, w in left:
        hit, best = None, WALL_TOLERANCE_S
        for s, wf in free:
            if abs(w - wf) <= best:
                hit, best = s, abs(w - wf)
        if hit is not None:
            hit.pose = p
            free = [(s, wf) for s, wf in free if s is not hit]


def unpaired_poses(root=None):
    """Camera CSVs with no timestamp, so nothing to attach them to."""
    root = Path(root).expanduser() if root else DATA_ROOT
    return sorted(p for p in _track_files(root)
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
    A run of digits means whichever of these its length implies:

        8   a date, MMDDYYYY or YYYYMMDD      07232026
        6   a time of day, HHMMSS             114556
        4   a month and day, any year         0723
        <=3 an index into what `ls` printed   2

    So:

        latest                  the newest session
        all                     everything
        07232026                every session on 23 Jul 2026
        07232026.114556         one run, named in full
        07232026.cam            ...that day's runs with camera data
        07232026.cam.last       ...the last of those
        07232026.2              the third of them, numbered as `ls` prints
        latest.real             newest session that is not SITL

    Filters: cam, nocam, sim, real. Ordinals: first, last, or an index.
    A bare 4-digit day is a convenience and is not year-qualified; prefer
    MMDDYYYY on a project that runs across years.
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
        elif token.isdigit() and len(token) == 4:
            pool = [s for s in pool if s.id[4:8] == token]      # MMDD
        elif token.isdigit() and len(token) == 6:
            pool = [s for s in pool if s.id[9:] == token]       # HHMMSS
        elif token.isdigit() and len(token) == 8:
            day = _as_date(token)
            if day is None:
                raise LookupError(
                    f"{token!r} is not a date; write it MMDDYYYY, "
                    f"e.g. 07232026 for 23 Jul 2026")
            pool = [s for s in pool if s.id.startswith(day)]
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
            f"{selector!r} matches {
                len(hits)} sessions; narrow it with a time "
            f"(e.g. 07232026.114556), .last, .first, or an index:"
            f"\n  {names}{more}")
    return hits[0]


# ----------------------------------------------------------------------------
def format_table(pool):
    """The `ls` listing: one line per session, newest last."""
    head = (f"{'#':>3}  {'session':16}  {'dur':>7}  {'rows':>6}  {'alt':>6}  "
            f"{'ref':>7}  {'flags':7}  where")
    lines = [head, "-" * len(head)]
    for i, s in enumerate(pool):
        d = s.summary
        where = str(s.flight.parent.relative_to(DATA_ROOT))
        dup = f"  [+{len(s.duplicates)} dup]" if s.duplicates else ""
        # a log with no rows is an aborted run; say so where mode used to
        flags = d["modes"] if d["modes"] in (
            "empty", "unreadable") else s.flags
        lines.append(
            f"{i:>3}  {s.id:16}  {d['duration']:6.1f}s  {d['rows']:>6}  "
            f"{d['alt']:6.1f}  {d['ref']:>7}  {flags:7}  {where}{dup}")
    return "\n".join(lines)
