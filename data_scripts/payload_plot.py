import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from estimation.calculate_payload_position import get_payload_ENU_from_data

G = 9.80665
MG_TO_MS2 = G/1000

CMD_TO_ENU = {"East (x)": ("ux", "aE"),
              "North (y)": ("uy", "aN"),
              "Up (z)": ("uz", "aU")}


MODE_SHADING = {
    "GUIDED": ("#4c72b0", 0.10),  # faint blue
    "STABILIZE": ("0.5", 0.12),  # faint gray
    "LOITER": ("#55a868", 0.14),  # faint green
}


def make_df(path):
    return pd.read_csv(os.path.expanduser(path))


def _set_equal_3d(ax, X, Y, Z):
    """Equal data aspect for a 3D axes (matplotlib has no axis('equal') in 3D)."""
    xr, yr, zr = np.ptp(X), np.ptp(Y), np.ptp(Z)
    r = max(xr, yr, zr) / 2 or 1.0
    cx, cy, cz = (X.max()+X.min())/2, (Y.max()+Y.min())/2, (Z.max()+Z.min())/2
    ax.set_xlim(cx-r, cx+r)
    ax.set_ylim(cy-r, cy+r)
    ax.set_zlim(cz-r, cz+r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _mode_runs(df):
    """
    Yield (start_idx, end_idx_inclusive, mode) for contiguous echoed_mode runs
    """
    em = df["echoed_mode"].astype(str).to_numpy()
    n = len(em)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and em[j + 1] == em[i]:
            j += 1
        yield i, j, em[i]
        i = j + 1


def trajectory_plot_3d_with_payload(df, payload_df, save=None, tether_every=25):
    """
    3D East-North-Up trajectory of the drone and its slung payload, colored by
    flight mode, with the reference trajectory overlaid. Gaps where the markers
    were not detected are left as breaks in the payload line.
    """
    E = df["drone_px_meas"].to_numpy()
    N = df["drone_py_meas"].to_numpy()
    U = df["drone_pz_meas"].to_numpy()

    pE = payload_df["payload_e"].to_numpy()
    pN = payload_df["payload_n"].to_numpy()
    pU = payload_df["payload_u"].to_numpy()

    fig = plt.figure(figsize=(8.5, 8))
    ax = fig.add_subplot(111, projection="3d")

    seen = set()
    for i, j, mode in _mode_runs(df):
        color = MODE_SHADING.get(mode, ("0.7", 0))[
            0]     # neutral gray if unknown
        # +1 pt to connect segments
        sl = slice(i, min(j + 2, len(df)))
        lbl = f"drone ({mode})" if mode not in seen else None
        ax.plot(E[sl], N[sl], U[sl], color=color, lw=1.8, label=lbl)
        seen.add(mode)

    if {"drone_px_ref", "drone_py_ref", "drone_pz_ref"} <= set(df.columns):
        ax.plot(df["drone_px_ref"], df["drone_py_ref"], df["drone_pz_ref"],
                "k--", lw=1.5, label="reference")

    # tether lines first so the payload trace draws on top of them
    finite = np.flatnonzero(np.isfinite(pE))
    if tether_every and finite.size:
        for k in finite[::tether_every]:
            ax.plot([E[k], pE[k]], [N[k], pN[k]], [U[k], pU[k]],
                    color="0.6", lw=0.6, alpha=0.7)

    ax.plot(pE, pN, pU, color="tab:orange", lw=1.2, label="payload")

    ax.scatter([E[0]], [N[0]], [U[0]], c="red", s=45, label="start")
    ax.scatter([0], [0], [0], c="black", marker="x",
               s=70, label="local origin")

    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Up [m]")
    ax.set_title("3D trajectory (ENU), drone and payload")
    _set_equal_3d(ax,
                  np.concatenate([E, pE[finite], [0]]),
                  np.concatenate([N, pN[finite], [0]]),
                  np.concatenate([U, pU[finite], [0]]))
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110)


if __name__ == "__main__":
    payload_pose_file = "~/TARES_SITL/data/test_07232026/all_data_07232026/step_test_with_camera_20260723_114555/poses.csv"
    flight_data_file = "~/TARES_SITL/data/test_07232026/all_data_07232026/flight_20260723_114556.csv"

    payload_df = get_payload_ENU_from_data(
        payload_pose_file, flight_data_file, time_offset=1)
    drone_df = pd.read_csv(flight_data_file)

    trajectory_plot_3d_with_payload(drone_df, payload_df)
    plt.show()
