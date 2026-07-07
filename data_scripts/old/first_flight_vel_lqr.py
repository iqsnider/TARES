"""Velocity tracking for the successful LQR trajectory tests."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
TRAJ_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]
DATA_DIR = os.path.expanduser("~/TARES_SITL/data/data_06302026/data/")

TRAJ = [
    ("flight_20260630_162306.csv", "16:23 traj LQR", "ok"),
    ("flight_20260630_162818.csv", "16:28 traj LQR", "ok"),
]
COLOR = {"dead": "#d62728", "displaced": "#ff7f0e",
         "on-ground": "#9467bd", "ok": "#2ca02c"}


def load(fname):
    return pd.read_csv(os.path.join(DATA_DIR, fname))


def main():
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: speed magnitude, actual (solid) vs reference (dotted)
    a = ax[0, 0]
    for i, (fname, lab, out) in enumerate(TRAJ):
        df = load(fname)

        color = TRAJ_COLORS[i % len(TRAJ_COLORS)]

        spd = np.sqrt(df.vx**2 + df.vy**2 + df.vz**2)
        spd_ref = np.sqrt(df.vx_ref**2 + df.vy_ref**2 + df.vz_ref**2)

        a.plot(df.t, spd, lw=1.8, color=color, label=lab)
        a.plot(df.t, spd_ref, ls=":", lw=1.2, color=color)

    a.set_title("Speed |v|: actual (solid) vs reference (dotted)")
    a.set_xlabel("t [s]")
    a.set_ylabel("speed [m/s]")
    a.legend(fontsize=8)
    a.grid(alpha=.3)

    axis_panels = {"vx": ax[0, 1], "vy": ax[1, 0], "vz": ax[1, 1]}
    for i, (axis, a) in enumerate(axis_panels.items()):
        for j, (fname, lab, out) in enumerate(TRAJ):
            df = load(fname)

            color = TRAJ_COLORS[j % len(TRAJ_COLORS)]

            a.plot(df.t, df[axis], lw=1.6, color=color, label=lab)
            a.plot(df.t, df[axis + "_ref"], ls=":", lw=1.2, color=color)

        a.set_title(f"{axis} tracking: actual (solid) vs ref (dotted)")
        a.set_xlabel("t [s]")
        a.set_ylabel(f"{axis} [m/s]")
        a.grid(alpha=.3)

        a.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(DATA_DIR, "traj_velocities.png")
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
