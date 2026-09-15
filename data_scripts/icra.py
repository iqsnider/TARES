"""
The paper's comparison figure: payload tracks from both controllers over one
reference square.

    uv run data_scripts/icra.py            show it
    uv run data_scripts/icra.py figs/      write it there

Both flight plans start at the northeast corner and go south, but the
ardupilot square logs its first corner as its first reference while the
payload square logs the start point, so runs are aligned on the reference's
northeast corner rather than on its first sample.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import catalog
from drone_plot import payload_enu
from paper_plot import _frame, ne_origin, C_INK, LEGEND_SIZE

C_PAYLOAD_CTRL = "#0A5CFF"   # bright blue
C_ARDUPILOT = "#E8112D"      # bright red

# the 20 m square, one run each, flown four minutes apart so the wind, the
# tether and the payload are the same in both
RUNS = [("09022026.141824", "ardupilot"),
        ("09022026.142232", "payload")]


def track(sel):
    """
    One run's payload track and reference, both put on the reference's
    northeast corner
    """
    s = catalog.resolve(sel)
    df = s.fl
    ref = df[["payload_px_ref", "payload_py_ref"]].to_numpy(float)
    origin = ne_origin(ref[np.isfinite(ref).all(1)])

    p, _ = payload_enu(df, L=s.config.get("TETHER_LEN"))

    return p[:, 0:2] - origin, ref - origin


def errors(sel):
    """
    Payload position error against the reference the run was chasing, RMS
    """
    p, ref = track(sel)
    track_err = np.hypot(*(p - ref).T)

    return np.sqrt(np.nanmean(track_err**2))


def rmse_figure(save=None):
    """
    What each controller cost, side by side
    """
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    names = {"ardupilot": "ArduPilot", "payload": "Payload"}
    colors = {"ardupilot": C_ARDUPILOT, "payload": C_PAYLOAD_CTRL}

    for x, (sel, kind) in enumerate(RUNS):
        ax.bar(x, errors(sel), width=0.6, color=colors[kind],
               edgecolor=C_INK, linewidth=0.9)
    ax.set_xticks(range(len(RUNS)))
    ax.set_xticklabels([names[kind] for _, kind in RUNS])
    ax.set_ylabel(r"Track error RMS [m]")
    ax.grid(axis="x", visible=False)
    _frame(ax)

    fig.tight_layout()
    if save:
        save = Path(save).expanduser()
        save.mkdir(parents=True, exist_ok=True)
        fig.savefig(save/"icra_payload_rmse.pdf", dpi=300, facecolor="white")

    return fig


def figure(save=None):
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    refs = {}
    seen = set()

    for sel, kind in RUNS:
        p, ref = track(sel)
        color = C_ARDUPILOT if kind == "ardupilot" else C_PAYLOAD_CTRL
        label = None if kind in seen else (
            "ArduPilot position control" if kind == "ardupilot"
            else "Payload control")
        seen.add(kind)
        ax.plot(p[:, 0], p[:, 1], color=color, lw=1.4, alpha=0.9, label=label)
        refs[round(abs(ref[:, 0].min()))] = ref

    # one dashed reference per square size flown
    for k, (edge, ref) in enumerate(sorted(refs.items())):
        ax.plot(ref[:, 0], ref[:, 1], ls=(0, (6, 4)), lw=1.5, color="#8A8F94",
                zorder=0, label="reference" if k == 0 else None)

    ax.set_xlabel(r"East [m]")
    ax.set_ylabel(r"North [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.06)
    _frame(ax)
    ax.legend(loc="best", framealpha=1.0, edgecolor=C_INK, facecolor="white",
              fancybox=False, fontsize=LEGEND_SIZE, labelcolor=C_INK)

    fig.tight_layout()
    if save:
        save = Path(save).expanduser()
        save.mkdir(parents=True, exist_ok=True)
        fig.savefig(save/"icra_payload_tracks.pdf", dpi=300, facecolor="white")

    return fig


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else None
    figure(out)
    rmse_figure(out)
    plt.show()
