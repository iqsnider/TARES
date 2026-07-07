"""Graph hardware flight tests. Edit DATA_DIR and the ACCEL/TRAJ lists per session."""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = os.path.expanduser("~/TARES_SITL/data/data_06302026/data/")

ACCEL = [
    ("smoke_20260630_151037.csv", "15:10 accel", "dead"),
    ("smoke_20260630_151202.csv", "15:12 accel", "dead"),
    ("smoke_20260630_160357.csv", "16:03 accel", "ok"),
]
TRAJ = [
    ("flight_20260630_162306.csv", "16:23 traj LQR", "ok"),
    ("flight_20260630_162818.csv", "16:28 traj LQR", "ok"),
]
COLOR = {"dead": "#d62728", "displaced": "#ff7f0e",
         "on-ground": "#9467bd", "ok": "#2ca02c"}

TRAJ_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def load(fname):
    return pd.read_csv(os.path.join(DATA_DIR, fname))


def main():
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))

    # Panel 1: accel smoke tests -- climb rate vs time (color by outcome, repeats OK)
    a = ax[0, 0]
    for fname, lab, out in ACCEL:
        df = load(fname)
        slope = np.polyfit(df["t"], df["vz_enu"], 1)[0]  # achieved accel
        a.plot(df["t"], df["vz_enu"], lw=2, marker=".", ms=4,
               color=COLOR[out], label=f"{lab}  (az\u2248{slope:.2f} m/s\u00b2)")
    a.set_title("Accel tests: commanded 1 m/s\u00b2 up\n"
                "flat = command ignored")
    a.set_xlabel("t [s]"); a.set_ylabel("climb rate vz [m/s]")
    a.legend(fontsize=9); a.grid(alpha=.3)

    # Panel 2: trajectory XY paths (actual solid, reference dotted)
    a = ax[0, 1]
    for i, (fname, lab, out) in enumerate(TRAJ):
        df = load(fname)

        color = TRAJ_COLORS[i % len(TRAJ_COLORS)]

        # Actual trajectory
        a.plot(
            df["px"], df["py"],
            lw=1.8,
            color=color,
            label=lab,
        )

        # Start marker (same color)
        a.plot(
            df["px"].iloc[0],
            df["py"].iloc[0],
            marker="o",
            ms=7,
            color=color,
            linestyle="None",
        )

        # Reference trajectory (same color, dotted)
        a.plot(
            df["px_ref"],
            df["py_ref"],
            ls=":",
            lw=1.5,
            color=color,
        )

    # Origin / home
    a.plot(
        0, 0,
        marker="o",
        color="k",
        ms=8,
        linestyle="None",
        label="origin/home",
    )

    a.set_title("Trajectory tests: XY path\ndotted = reference")
    a.set_xlabel("px (E) [m]")
    a.set_ylabel("py (N) [m]")
    a.legend(fontsize=8)
    a.grid(alpha=.3)

    # Panel 3: trajectory tracking error
    a = ax[1, 0]
    for i, (fname, lab, out) in enumerate(TRAJ):
        df = load(fname)
        a.plot(df["t"], df["pos_err_norm"], lw=1.8,
               label=f"{lab}")
    a.axhline(0.5, color="g", ls="--", lw=1)
    a.set_title("Trajectory tests: position error")
    a.set_xlabel("t [s]"); a.set_ylabel("pos err norm [m]")
    a.legend(fontsize=8); a.grid(alpha=.3)

    # Panel 4: trajectory control effort
    a = ax[1, 1]
    for i, (fname, lab, out) in enumerate(TRAJ):
        df = load(fname)
        a.plot(df["t"], df["u_norm"], lw=1.8, label=f"{lab}")
    a.set_title("Trajectory tests: control effort |u|\n")
    a.set_xlabel("t [s]"); a.set_ylabel("u_norm [m/s\u00b2]")
    a.legend(fontsize=8); a.grid(alpha=.3)

    plt.tight_layout()
    out_path = os.path.join(DATA_DIR, "hardware_tests.png")
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
