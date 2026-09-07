"""
The two figures for the paper.

Separate from drone_plot because they are built to be read on a printed page
rather than to diagnose a flight: no mode shading, a full frame, faint grid,
and nothing on them that a caption would not explain.
"""

import numpy as np
import matplotlib.pyplot as plt

from drone_plot import payload_enu, _has
from sim.plotting import configure_plot_style

configure_plot_style()   # shared serif / Computer Modern theme

# the drone plot's velocity colors, brightened for print
C_E = "#0C6BE8"
C_N = "#F97306"
C_U = "#12A150"
C_PAYLOAD_PATH = "#0A5CFF"   # bright blue, the payload-control color
C_REF_LINE = "#8A8F94"

# frame, ticks and text: black for print, sized to survive a two column figure
C_INK = "#000000"
TICK_SIZE = 12.5
LEGEND_SIZE = 12.5

REF_SETS = [("drone", "drone_v{}_ref"), ("payload", "payload_v{}_ref")]


def _frame(ax):
    """
    Boxed axes, faint grid, no title: the same figure in every paper
    """
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.1)
        ax.spines[side].set_color(C_INK)

    ax.grid(True, color="#B8B8B8", lw=0.5, alpha=0.35)
    ax.set_axisbelow(True)
    ax.tick_params(direction="out", width=1.1, color=C_INK,
                   labelcolor=C_INK, labelsize=TICK_SIZE)
    ax.xaxis.label.set_color(C_INK)
    ax.yaxis.label.set_color(C_INK)
    ax.set_facecolor("white")
    ax.figure.set_facecolor("white")


def ne_origin(ref):
    """
    The reference square's northeast corner, which is where both flight plans
    start before heading south. Aligning on it rather than on the first logged
    sample matters because the ardupilot square never logs its start point.
    """
    return np.array([ref[:, 0].max(), ref[:, 1].max()])


def _velocity_reference(df):
    """
    Whichever reference the run actually flew, drone or payload, and its name
    """
    for name, pattern in REF_SETS:
        cols = [pattern.format(a) for a in ("x", "y", "z")]
        if _has(df, *cols) and df[cols].notna().any().any():
            return name, cols

    return None, None


def velocity_reference_plot(df, save=None):
    """
    Figure 1: the velocity the mission asked for, on its own
    """
    name, cols = _velocity_reference(df)
    if cols is None:
        raise SystemExit("this log has no velocity reference to draw")

    t = df["cur_time"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.0, 3.2))

    for col, color, label in zip(cols, (C_E, C_N, C_U),
                                 (r"$v_E$", r"$v_N$", r"$v_U$")):
        ax.plot(t, df[col], color=color, lw=1.8, label=label)

    ax.set_xlabel(r"Time [s]")
    ax.set_ylabel(rf"{name.capitalize()} velocity reference [m/s]")
    ax.set_xlim(t[0], t[-1])
    _frame(ax)
    ax.legend(loc="best", framealpha=1.0, edgecolor=C_INK,
              facecolor="white", fancybox=False, fontsize=LEGEND_SIZE,
              labelcolor=C_INK)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, facecolor="white")

    return fig


def payload_trajectory_plot(df, save=None, L=None):
    """
    Figure 2: the payload in the plane, against the track it was asked to fly,
    with the origin put on the start of the reference
    """
    if not _has(df, "payload_px_ref", "payload_py_ref"):
        raise SystemExit("this log has no payload reference to draw")

    ref = df[["payload_px_ref", "payload_py_ref"]].dropna().to_numpy(float)
    origin = ne_origin(ref)
    ref = ref - origin

    p, _ = payload_enu(df, L=L)
    E, N = p[:, 0] - origin[0], p[:, 1] - origin[1]

    fig, ax = plt.subplots(figsize=(5.6, 5.6))

    ax.plot(ref[:, 0], ref[:, 1], ls=(0, (6, 4)), lw=1.5, color=C_REF_LINE,
            zorder=0, label=r"reference")
    ax.plot(E, N, color=C_PAYLOAD_PATH, lw=1.4, alpha=0.9, label=r"payload")

    ax.set_xlabel(r"East [m]")
    ax.set_ylabel(r"North [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.06)
    _frame(ax)
    ax.legend(loc="best", framealpha=1.0, edgecolor=C_INK,
              facecolor="white", fancybox=False, fontsize=LEGEND_SIZE,
              labelcolor=C_INK)

    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=300, facecolor="white")

    return fig


def plot_suite(df, save=None, stem="flight", L=None):
    """
    Both figures, written as <stem>_velocity_reference and <stem>_payload_track
    """
    if save:
        save.mkdir(parents=True, exist_ok=True)

    velocity_reference_plot(
        df, save=save and save/f"{stem}_velocity_reference.pdf")
    payload_trajectory_plot(
        df, save=save and save/f"{stem}_payload_track.pdf", L=L)
