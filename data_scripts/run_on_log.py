"""Run the payload EKF against a real flight log + pose CSV."""
import numpy as np
import pandas as pd
import math
import cv2
import sim.config as config
import estimation.calculate_payload_position as payload
import estimation.ekf as ekfm

G = 9.80665


def build_inputs(fl):
    """Inertial-ENU specific force + attitude vs flight time. Computed once."""
    a_frd = fl[['drone_ax_meas', 'drone_ay_meas', 'drone_az_meas']].to_numpy()*G / \
        1000.
    roll = fl.drone_roll.to_numpy()
    pitch = fl.drone_pitch.to_numpy()
    yaw = fl.drone_yaw.to_numpy()
    u = np.empty_like(a_frd)
    for i in range(len(fl)):
        u[i] = ekfm.C_nb_enu(roll[i], pitch[i], yaw[i]
                             ) @ (ekfm.P_SWAP @ a_frd[i])
    return dict(t=fl.cur_time.to_numpy(), u=u, roll=roll, pitch=pitch, yaw=yaw)


def group_frames(pose):
    """[(t_cam, [(id,u,v),...], first_det_row_or_None, frame_df), ...] sorted by time."""
    out = []
    for _, g in pose.groupby('frame'):
        det = g.dropna(subset=['marker_id'])
        meas = [(int(m.marker_id), m.u_px, m.v_px) for m in det.itertuples()]
        out.append((float(g.time_s.iloc[0]), meas, det, g))
    out.sort(key=lambda kv: kv[0])
    return out


def run(frames, inp, params, offset):
    t_fl, u_fl = inp['t'], inp['u']
    xi = P = None
    t_prev = None
    rows = []
    for t_cam, meas, det, g in frames:
        t = t_cam - offset
        if t < t_fl[0] or t > t_fl[-1]:
            continue
        r = np.interp(t, t_fl, inp['roll'])
        p_ = np.interp(t, t_fl, inp['pitch'])
        y = np.interp(t, t_fl, inp['yaw'])
        uu = np.array([np.interp(t, t_fl, u_fl[:, 0]),
                       np.interp(t, t_fl, u_fl[:, 1]),
                       np.interp(t, t_fl, u_fl[:, 2])])
        if xi is None:
            if not meas:
                continue
            ctr = payload.get_payload_center_in_camera_frame(g)
            R, _ = cv2.Rodrigues(det[['rx', 'ry', 'rz']].iloc[0].to_numpy())
            xi, P = ekfm.initial_state(ctr, r, p_, y, marker_R=R)
            t_prev = t_cam
            continue
        dt = min(max(t_cam - t_prev, 1e-3), 0.5)
        t_prev = t_cam
        xi, P, info = ekfm.ekf(xi, P, uu, dt, params, measurements=meas,
                               roll=r, pitch=p_, yaw=y)
        rows.append((t_cam, t, info['n_markers'], info['nis'],
                     xi[0], xi[1], xi[2], xi[3], xi[4],
                     math.sqrt(P[0, 0]), math.sqrt(P[1, 1]), r, p_, y))
    return pd.DataFrame(rows, columns=['t_cam', 't_flight', 'n', 'nis', 'ax', 'ay',
                                       'dax', 'day', 'yawp', 'sax', 'say',
                                       'roll', 'pitch', 'yaw'])


def mean_nnis(df, warmup=3.0):
    if len(df) == 0:
        return np.inf
    m = df[(df.n > 0) & (df.t_cam > df.t_cam.min() + warmup)]
    if len(m) < 20:
        return np.inf
    return float(np.mean(m.nis.to_numpy()/(2*m.n.to_numpy())))


CALIB_PATH = "~/TARES_SITL/src/payload_tracking/camera_calibration/calibration.json"
_CALIB_CACHE = {}


def load_calibration(path=CALIB_PATH):
    """Read calibration.json -> (mtx 3x3, dist 1-D). Cached, '~' expanded."""
    import json
    import os
    path = os.path.expanduser(str(path))
    if path not in _CALIB_CACHE:
        with open(path) as f:
            c = json.load(f)
        mtx = np.asarray(c["mtx"], dtype=float)
        dist = np.asarray(c["dist"], dtype=float).ravel()
        _CALIB_CACHE[path] = (mtx, dist)
    return _CALIB_CACHE[path]


def make_params(L, L_m=8.31, calib_path=CALIB_PATH, **kw):
    mtx, dist = load_calibration(calib_path)
    kw.setdefault("q", 0.02**2)
    kw.setdefault("q_yaw", 0.3**2)
    kw.setdefault("sigma_det", 0.4)
    kw.setdefault("sigma_att", math.radians(0.5))
    return ekfm.EKFParams(mtx, dist, L=L, L_m=L_m, **kw)


if __name__ == "__main__":
    import sys
    import time
    pose = pd.read_csv('/mnt/user-data/uploads/poses.csv')
    fl = pd.read_csv('/mnt/user-data/uploads/flight_20260723_114556.csv')
    inp = build_inputs(fl)
    frames = group_frames(pose)
    t0 = time.time()
    offs = np.arange(-1.0, 2.6, 0.10)
    Ls = np.array([6.5, 7.0, 7.4, 7.8, 8.3])
    table = np.full((len(Ls), len(offs)), np.inf)
    for i, L in enumerate(Ls):
        prm = make_params(L)
        for j, o in enumerate(offs):
            table[i, j] = mean_nnis(run(frames, inp, prm, o))
    print(f"({time.time()-t0:.0f}s)\n{'L_eff':>7} | best offset    min NIS")
    for i, L in enumerate(Ls):
        j = int(np.argmin(table[i]))
        print(f"{L:>7.2f} | {offs[j]:>+10.2f} s {table[i, j]:>10.2f}")
    i, j = np.unravel_index(np.argmin(table), table.shape)
    print(f"\nBEST: offset {
          offs[j]:+.2f} s, L_eff {Ls[i]:.2f} m, NIS {table[i, j]:.2f}")
