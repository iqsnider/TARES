import argparse
import os

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt


G = 9.80665
MG_TO_MS2 = G/1000

CMD_TO_ENU = {"East (x)":  ("ux", "aE"),
              "North (y)": ("uy", "aN"),
              "Up (z)":    ("uz", "aU")}


def make_df(path):
    return pd.read_csv(os.path.expanduser(path))


def imu_update_rate(df):
    """Effective RAW_IMU update rate, inferred from how often the value changes.

    Useful sanity check: if this is well below your loop rate, the RAW_SENSORS
    stream isn't being requested fast enough and the measured trace will staircase.
    """
    t = df["cur_time"].to_numpy()
    a = df["drone_ay_meas"].to_numpy()
    idx = np.flatnonzero(np.diff(a) != 0) + 1
    if len(idx) < 2:
        return float("nan")
    return 1.0 / np.mean(np.diff(t[idx]))


def measured_accel_enu(df, g=G):
    """Reconstruct inertial acceleration in ENU from body-frame specific force.

    Steps: raw mg -> m/s^2, rotate body(FRD)->NED with the aerospace 3-2-1 Euler
    angles, remove gravity, then reorder NED->ENU.  Returns (aE, aN, aU) arrays;
    rows with missing attitude/accel come back as NaN (matplotlib skips them).
    """
    fx = df["drone_ax_meas"].to_numpy()*MG_TO_MS2
    fy = df["drone_ay_meas"].to_numpy()*MG_TO_MS2
    fz = df["drone_az_meas"].to_numpy()* MG_TO_MS2

    roll = df["drone_roll"].to_numpy()
    pitch = df["drone_pitch"].to_numpy()
    yaw = df["drone_yaw"].to_numpy()

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    # R_body->NED * f_body
    n = (cp * cy) * fx + (sr * sp * cy - cr * sy) * fy + (cr * sp * cy + sr * sy) * fz
    e = (cp * sy) * fx + (sr * sp * sy + cr * cy) * fy + (cr * sp * sy - sr * cy) * fz
    d = (-sp) * fx + (sr * cp) * fy + (cr * cp) * fz

    # inertial acceleration = specific force + gravity([0,0,+g] in NED)
    aN = n
    aE = e
    aU = -(d + g)
    return aE, aN, aU


def acc_plot(df):
    """Three-panel command-vs-measured acceleration comparison, saved to out_path."""
    t = df["cur_time"].to_numpy()
    aE, aN, aU = measured_accel_enu(df)
    recon = {"aE": aE, "aN": aN, "aU": aU}


    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)

    for ax, (label, (cmd_col, enu_key)) in zip(axes, CMD_TO_ENU.items()):

        ax.plot(t, df[cmd_col], label=f"commanded {cmd_col} ENU")
        ax.plot(t, recon[enu_key], label="measured (IMU -> ENU, gravity removed)")
        ax.set_ylabel(f"{label}\naccel [m/s$^2$]")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time [s]")
    plt.show()



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare commanded vs IMU-measured acceleration.")
    ap.add_argument("csv", nargs="?",
                    default="~/TARES_SITL/data/flight_20260713_113725.csv",
                    help="path to a flight CSV log")
    args = ap.parse_args()

    df = make_df(args.csv)
    acc_plot(df)
