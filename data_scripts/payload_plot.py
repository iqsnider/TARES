import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# stable color per marker id
MARKER_COLORS = {231: "#4c72b0", 232: "#dd8452",
                 233: "#55a868", 234: "#c44e52", 235: "#8172b3",
                 230: "#937860"}


def load_poses(path):
    p = pd.read_csv(os.path.expanduser(path))
    p = p.dropna(subset=["marker_id", "x", "y", "z"]).copy()
    p["marker_id"] = p["marker_id"].astype(int)
    return p


def load_flight(path):
    return pd.read_csv(os.path.expanduser(path)) if path else None


# ---------- 1. per-marker XY trajectory (camera frame) ----------
def marker_xy_plot(poses, save=None):
    fig, ax = plt.subplots(figsize=(8, 7))
    for mid, g in poses.groupby("marker_id"):
        g = g.sort_values("time_s")
        c = MARKER_COLORS.get(int(mid), None)
        ax.plot(g.x, g.y, "-", lw=0.8, alpha=0.5, color=c)
        ax.scatter(g.x, g.y, s=8, color=c, label=f"id {int(mid)} (n={len(g)})")
    ax.set_xlabel("x [m] (camera frame)")
    ax.set_ylabel("y [m] (camera frame)")
    ax.set_title("Per-marker XY trajectory (camera frame)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110)


# ---------- payload center + camera->ENU registration ----------
def payload_center_enu(poses, flight):
    """
    One payload point per camera frame (centroid of visible markers), registered
    into the drone's ENU frame by timestamp.

    >>> FRAME ASSUMPTIONS (edit here) <<<
    - Camera rigidly mounted on the drone, pointing straight down (nadir).
    - OpenCV optical frame: +z forward (down) => world U offset = -z.
    - Horizontal (x,y) placed WITHOUT yaw/mounting correction (baseline).
      Vertical is reliable; horizontal E/N may need a yaw rotation + axis/sign
      fix once the camera clocking is known.
    """
    per_frame = (poses.groupby("frame")
                 .agg(time_s=("time_s", "mean"),
                      x=("x", "mean"), y=("y", "mean"), z=("z", "mean"),
                      n=("marker_id", "size"))
                 .reset_index()
                 .sort_values("time_s"))

    fl = flight.sort_values("cur_time")
    merged = pd.merge_asof(per_frame, fl[["cur_time", "drone_px_meas",
                                          "drone_py_meas", "drone_pz_meas",
                                          "drone_yaw"]],
                           left_on="time_s", right_on="cur_time",
                           direction="nearest")

    # --- editable transform block ---
    merged["payload_E"] = merged.drone_px_meas + merged.x
    merged["payload_N"] = merged.drone_py_meas + merged.y
    merged["payload_U"] = merged.drone_pz_meas - merged.z
    return merged


# ---------- 2. drone + payload XYZ trajectory ----------
def drone_payload_xyz_plot(poses, flight, save=None):
    m = payload_center_enu(poses, flight)
    fl = flight
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(fl.drone_px_meas, fl.drone_py_meas, fl.drone_pz_meas,
            color="#4c72b0", lw=1.5, label="drone (ENU)")
    ax.scatter(fl.drone_px_meas.iloc[0], fl.drone_py_meas.iloc[0],
               fl.drone_pz_meas.iloc[0], color="#4c72b0", s=40)

    ax.plot(m.payload_E, m.payload_N, m.payload_U,
            color="#c44e52", lw=1.0, alpha=0.8, label="payload (centroid)")

    # faint vertical tether lines every ~2 s to show hang geometry
    step = max(1, len(m) // 40)
    for _, r in m.iloc[::step].iterrows():
        di = (fl.cur_time - r.time_s).abs().idxmin()
        ax.plot([fl.drone_px_meas[di], r.payload_E],
                [fl.drone_py_meas[di], r.payload_N],
                [fl.drone_pz_meas[di], r.payload_U],
                color="0.6", lw=0.4, alpha=0.4)

    ax.set_xlabel("E [m]"); ax.set_ylabel("N [m]"); ax.set_zlabel("U [m]")
    ax.set_title("Drone + payload trajectory (ENU)\n"
                 "payload horizontal = uncorrected for yaw/camera clocking")
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Payload pose analysis plots.")
    ap.add_argument("poses", nargs="?",
                    default="~/TARES_SITL/poses_170mm_test2_iteration_0.csv")
    ap.add_argument("--flight", default="~/TARES_SITL/camera_test_data/flight_20260715_151404.csv",
                    help="flight CSV for the drone trajectory (graph 2)")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    poses = load_poses(args.poses)
    flight = load_flight(args.flight)

    ids = sorted(poses.marker_id.unique())
    print(f"marker ids present: {[int(i) for i in ids]}  "
          f"({len(ids)} of 5 expected)")

    def out(name):
        if args.outdir:
            base = os.path.splitext(os.path.basename(args.poses))[0]
            return os.path.join(os.path.expanduser(args.outdir), f"{base}_{name}.png")
        return None

    marker_xy_plot(poses, save=out("marker_xy"))
    if flight is not None:
        drone_payload_xyz_plot(poses, flight, save=out("drone_payload_xyz"))
    else:
        print("no --flight given; skipping drone+payload XYZ plot")

    if not args.outdir:
        plt.show()
