import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

G = 9.80665
MG_TO_MS2 = G/1000

CMD_TO_ENU = {"East (x)": ("ux", "aE"),
              "North (y)": ("uy", "aN"),
              "Up (z)": ("uz", "aU")}

GUIDED_MODES = {"GUIDED"}

MODE_SHADING = {
    "GUIDED": ("#4c72b0", 0.10), # faint blue
    "STABILIZE": ("0.5", 0.12),  # faint gray
    "LOITER": ("#55a868", 0.14), # faint green
}


def make_df(path):
    return pd.read_csv(os.path.expanduser(path))


def guided_window(df):
    """
    Finds the time of the manual takeover
    """
    em = df["echoed_mode"].astype(str)
    is_guided = em.isin(GUIDED_MODES)
    if not is_guided.any():
        # never confirmed guided; keep everything, warn caller
        return None, pd.Series(True, index=df.index)

    first_guided = is_guided.idxmax()  # first True
    # takeover = first real (non-"?", non-NaN) mode != GUIDED after guided starts
    after = df.index > first_guided
    left_guided = after & ~is_guided & ~em.isin(["?", "nan", "None"])
    if not left_guided.any():
        return None, pd.Series(True, index=df.index)

    takeover_idx = left_guided.idxmax()
    t_takeover = float(df["cur_time"].iloc[takeover_idx])
    mask = df["cur_time"] < t_takeover
    return t_takeover, mask


def measured_accel_enu(df, g=G):
    """
    Reconstruct inertial acceleration in ENU from body-frame specific force.
    """
    fx = df["drone_ax_meas"].to_numpy()*MG_TO_MS2
    fy = df["drone_ay_meas"].to_numpy()*MG_TO_MS2
    fz = df["drone_az_meas"].to_numpy()*MG_TO_MS2

    roll = df["drone_roll"].to_numpy()
    pitch = df["drone_pitch"].to_numpy()
    yaw = df["drone_yaw"].to_numpy()

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    T_EB = np.array([[cp*cy, sr*sp*cy - cr*sy, cr*sp*cy + sr*sy],
                     [cp*sy, sr*sp*sy + cr*cy, cr*sp*sy - sr*cy],
                     [-sp,   sr*cp,            cr*cp]])
    T_EB = np.moveaxis(T_EB, 2, 0)

    f_body = np.stack([fx, fy, fz], axis=1)
    ned = T_EB @ f_body[..., None]

    n, e, d = ned[:, 0, 0], ned[:, 1, 0], ned[:, 2, 0]

    return e, n, -(d + g) # aE, aN, aU



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


def _shade_modes(ax, df, label=True):
    """
    Shade mode regions on plot
    """
    em = df["echoed_mode"].astype(str).to_numpy()
    t = df["cur_time"].to_numpy()
    n = len(em)
    seen = set()
    i = 0
    while i < n:
        j = i
        while j + 1 < n and em[j + 1] == em[i]:
            j += 1
        mode = em[i]
        if mode in MODE_SHADING:
            color, alpha = MODE_SHADING[mode]
            t0 = t[i]
            t1 = t[j + 1] if j + 1 < n else t[j]     # extend to next segment
            lbl = mode if (label and mode not in seen) else None
            ax.axvspan(t0, t1, color=color, alpha=alpha, lw=0, label=lbl)
            seen.add(mode)
        # dashed line at the boundary into the next segment (i.e. a mode change)
        if i > 0:
            ax.axvline(t[i], color="k", ls="--", lw=1, alpha=0.6)
        i = j + 1


def acc_plot(df, mask, t_takeover, save=None):
    """
    Plot 1: Commanded and Measured acceleration vs time, cuts at manual takeover
    """
    d = df[mask]
    t = d["cur_time"].to_numpy()
    aE, aN, aU = measured_accel_enu(d)
    recon = {"aE": aE, "aN": aN, "aU": aU}

    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

    for k, (ax, (axis_label, (cmd_col, enu_key))) in enumerate(zip(axes, CMD_TO_ENU.items())):
        ax.plot(t, d[cmd_col], label=f"Commanded {cmd_col} ENU")
        ax.plot(t, recon[enu_key], label="Measured (IMU -> ENU, g removed)")
        _shade_modes(ax, d, label=(k == 0))
        ax.set_ylabel(f"{axis_label}\nAccel [m/s$^2$]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[0].set_title("Commanded vs Measured Acceleration (GUIDED only)")
    axes[-1].set_xlabel("Time [s]")
    
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=110)


def position_plot(df, t_takeover, save=None):
    """
    Plot 2: Drone position vs time
    """
    t = df["cur_time"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5))

    for comp, col, ref in [("E", "drone_px_meas", "drone_px_ref"),
                           ("N", "drone_py_meas", "drone_py_ref"),
                           ("U", "drone_pz_meas", "drone_pz_ref")]:

        line, = ax.plot(t, df[col], label=f"{comp} meas")

        if ref in df:
            ax.plot(t, df[ref], ls="--", lw=1, color=line.get_color(),
                    alpha=0.6, label=f"{comp} ref")

    ax.set_xlabel("Time [s]"); ax.set_ylabel("Position ENU [m]")
    ax.set_title("Drone Position vs Time")

    ax.grid(True, alpha=0.3)
    _shade_modes(ax, df)
    ax.legend(fontsize=8, ncol=3, loc="best")

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=110)


def velocity_plot(df, t_takeover, save=None):
    """
    Plot 3: Drone velocity vs time
    """
    t = df["cur_time"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5))

    for comp, col, ref in [("vE", "drone_vx_meas", "drone_vx_ref"),
                           ("vN", "drone_vy_meas", "drone_vy_ref"),
                           ("vU", "drone_vz_meas", "drone_vz_ref")]:

        line, = ax.plot(t, df[col], label=f"{comp} meas")

        if ref in df:
            ax.plot(t, df[ref], ls="--", lw=1, color=line.get_color(),
                    alpha=0.6, label=f"{comp} ref")

    ax.set_xlabel("Time [s]"); ax.set_ylabel("Velocity ENU [m/s]")
    ax.set_title("Drone Velocity vs Time")

    ax.grid(True, alpha=0.3)
    _shade_modes(ax, df)
    ax.legend(fontsize=8, ncol=3, loc="best")

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=110)


def attitude_plot(df, t_takeover, save=None):
    """
    Plot 4: Attitude vs time
    """
    t = df["cur_time"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(t, np.degrees(df["drone_roll"]),  label="roll")
    ax.plot(t, np.degrees(df["drone_pitch"]), label="pitch")
    ax.plot(t, np.degrees(df["drone_yaw"]),   label="yaw")

    ax.set_xlabel("Time [s]"); ax.set_ylabel("Angle [deg]")

    ax.set_title("Attitude vs Time")

    ax.grid(True, alpha=0.3)
    _shade_modes(ax, df)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110)


def ctrl_freq_plot(df, mask, target_hz=None, save=None):
    """
    Plot 5: Control frequency vs time
    """

    d = df[mask]
    t = d["cur_time"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(t, d["cur_ctrl_freq"], label="control freq (cur_ctrl_freq)")

    if target_hz:
        ax.axhline(target_hz, color="green", ls="--", lw=1,
                   label=f"target {target_hz} Hz")


    _shade_modes(ax, d)

    ax.set_xlabel("Time [s]"); ax.set_ylabel("Frequency [Hz]")
    ax.set_title("Control Loop Frequency vs Time (GUIDED only)")

    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=110)


def rc_plot(df, t_takeover, save=None):
    """
    Plot 6: RC channels vs time
    """
    t = df["cur_time"].to_numpy()

    fig, ax = plt.subplots(figsize=(11, 5.5))

    for ch in [f"ch{i}" for i in range(1, 9)]:
        if ch in df:
            ax.plot(t, df[ch], label=ch)

    ax.set_xlabel("Time [s]"); ax.set_ylabel("PWM")
    ax.set_title("RC Channels vs Time")

    ax.grid(True, alpha=0.3)
    _shade_modes(ax, df)
    ax.legend(fontsize=8, ncol=4, loc="best")

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=110)


def trajectory_plot(df, save=None):
    """
    Plot 7: East-North trajectory of drone, colored by flight mode.
    Reference trajectory is also shown.
    """
    E = df["drone_px_meas"].to_numpy()
    N = df["drone_py_meas"].to_numpy()

    fig, ax = plt.subplots(figsize=(7.5, 7.5))

    seen = set()
    for i, j, mode in _mode_runs(df):
        color = MODE_SHADING.get(mode, ("0.7", 0))[0] # neutral gray if unknown
        sl = slice(i, min(j + 2, len(df))) # +1 pt to connect segments
        lbl = f"drone ({mode})" if mode not in seen else None
        ax.plot(E[sl], N[sl], color=color, lw=1.8, label=lbl)
        seen.add(mode)

    if "drone_px_ref" in df and "drone_py_ref" in df:
        ax.plot(df["drone_px_ref"], df["drone_py_ref"], "k--", lw=1.5,
                label="reference")

    ax.scatter([E[0]], [N[0]], c="red", zorder=5, s=45, label="start")
    ax.scatter([0], [0], c="black", marker="x", s=70, zorder=5,
               label="local origin")

    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]")
    ax.set_title("Horizontal trajectory (ENU), colored by mode")

    ax.axis("equal"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=110)

 
def _set_equal_3d(ax, X, Y, Z):
    """Equal data aspect for a 3D axes (matplotlib has no axis('equal') in 3D)."""
    xr, yr, zr = np.ptp(X), np.ptp(Y), np.ptp(Z)
    r = max(xr, yr, zr) / 2 or 1.0
    cx, cy, cz = (X.max()+X.min())/2, (Y.max()+Y.min())/2, (Z.max()+Z.min())/2
    ax.set_xlim(cx-r, cx+r); ax.set_ylim(cy-r, cy+r); ax.set_zlim(cz-r, cz+r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

def trajectory_plot_3d(df, save=None):
    """
    3D East-North-Up trajectory of the drone, colored by flight mode, with the
    reference trajectory overlaid. 3D counterpart of trajectory_plot.
    """
    E = df["drone_px_meas"].to_numpy()
    N = df["drone_py_meas"].to_numpy()
    U = df["drone_pz_meas"].to_numpy()
    fig = plt.figure(figsize=(8.5, 8))
    ax = fig.add_subplot(111, projection="3d")
 
    seen = set()
    for i, j, mode in _mode_runs(df):
        color = MODE_SHADING.get(mode, ("0.7", 0))[0]     # neutral gray if unknown
        sl = slice(i, min(j + 2, len(df)))                # +1 pt to connect segments
        lbl = f"drone ({mode})" if mode not in seen else None
        ax.plot(E[sl], N[sl], U[sl], color=color, lw=1.8, label=lbl)
        seen.add(mode)
 
    if {"drone_px_ref", "drone_py_ref", "drone_pz_ref"} <= set(df.columns):
        ax.plot(df["drone_px_ref"], df["drone_py_ref"], df["drone_pz_ref"],
                "k--", lw=1.5, label="reference")
 
    ax.scatter([E[0]], [N[0]], [U[0]], c="red", s=45, label="start")
    ax.scatter([0], [0], [0], c="black", marker="x", s=70, label="local origin")
 
    ax.set_xlabel("East [m]"); ax.set_ylabel("North [m]"); ax.set_zlabel("Up [m]")
    ax.set_title("3D trajectory (ENU), colored by mode")
    _set_equal_3d(ax, np.append(E, 0), np.append(N, 0), np.append(U, 0))
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=110)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Post-flight analysis plots.")
    ap.add_argument("csv", nargs="?",
                    default="~/TARES_SITL/data/data_07152026/flight_20260715_133242.csv")
    ap.add_argument("--target-hz", type=float, default=50,
                    help="expected control frequency for the reference line")
    ap.add_argument("--outdir", default=None,
                    help="if set, save PNGs here instead of (only) showing")
    args = ap.parse_args()

    df = make_df(args.csv)
    t_takeover, mask = guided_window(df)
    if t_takeover is None:
        print("No manual takeover detected; using full flight.")
    else:
        print(f"Manual takeover (mode left GUIDED) at t = {t_takeover:.3f} s "
              f"({int(mask.sum())}/{len(df)} samples are GUIDED)")

    def out(name):
        if args.outdir:
            base = os.path.splitext(os.path.basename(args.csv))[0]
            return os.path.join(os.path.expanduser(args.outdir), f"{base}_{name}.png")
        return None

    acc_plot(df, mask, t_takeover, save=out("accel"))
    position_plot(df, t_takeover, save=out("position"))
    velocity_plot(df, t_takeover, save=out("velocity"))
    attitude_plot(df, t_takeover, save=out("attitude"))
    ctrl_freq_plot(df, mask, target_hz=args.target_hz, save=out("ctrlfreq"))
    rc_plot(df, t_takeover, save=out("rc"))
    trajectory_plot(df, save=out("trajectory"))
    trajectory_plot_3d(df, save=out("trajectory_3d"))

    if not args.outdir:
        plt.show()
