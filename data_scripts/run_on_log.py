"""Run the payload EKF against a real flight log + pose CSV."""
import numpy as np
import pandas as pd
import math
import cv2
import sim.estimation.calculate_payload_position as payload
import sim.estimation.ekf as ekfm
import catalog

G = 9.80665


def build_inputs(fl):
    """Inertial-ENU specific force f_I + attitude vs flight time. Computed once."""
    a_frd = fl[['drone_ax_meas', 'drone_ay_meas', 'drone_az_meas']].to_numpy()*G / \
        1000.
    phi = fl.drone_roll.to_numpy()
    theta = fl.drone_pitch.to_numpy()
    psi = fl.drone_yaw.to_numpy()
    f_I = np.empty_like(a_frd)
    for i in range(len(fl)):
        f_I[i] = ekfm.C_IB_enu(phi[i], theta[i], psi[i]
                               ) @ (ekfm.P_SWAP @ a_frd[i])
    return dict(t=fl.cur_time.to_numpy(), f_I=f_I, phi=phi, theta=theta, psi=psi)


def group_frames(pose):
    """[(t_cam, p_C_payload, C_CM, det), ...] sorted by time.

    This is the pre-processing the filter no longer does: the payload board
    center in the camera frame and the marker rotation, both out of PnP.
    p_C_payload and C_CM are None for a frame with no detection; `det` is the
    frame's detected rows, kept for the marker count and for plotting.
    """
    out = []
    for _, g in pose.groupby('frame'):
        det = g.dropna(subset=['marker_id'])
        p_C = payload.get_payload_center_in_camera_frame(g)
        C_CM = None
        if len(det):
            C_CM, _ = cv2.Rodrigues(det[['rx', 'ry', 'rz']].iloc[0].to_numpy())
        out.append((float(g.time_s.iloc[0]), p_C, C_CM, det))
    out.sort(key=lambda kv: kv[0])
    return out


def dof(C_CM):
    """Measurement rows folded in: 2 bearing, plus the yaw row when PnP gave one."""
    return ekfm.MEAS_DIM if C_CM is not None else 2


def run(frames, inp, params, offset):
    t_fl, f_fl = inp['t'], inp['f_I']
    xi = P = None
    t_prev = None
    rows = []
    for t_cam, p_C, C_CM, det in frames:
        t = t_cam - offset
        if t < t_fl[0] or t > t_fl[-1]:
            continue
        phi = np.interp(t, t_fl, inp['phi'])
        theta = np.interp(t, t_fl, inp['theta'])
        psi = np.interp(t, t_fl, inp['psi'])
        f_I = np.array([np.interp(t, t_fl, f_fl[:, 0]),
                        np.interp(t, t_fl, f_fl[:, 1]),
                        np.interp(t, t_fl, f_fl[:, 2])])
        if xi is None:
            if p_C is None:
                continue
            xi, P = ekfm.initial_state(p_C, phi, theta, psi, params, C_CM=C_CM)
            t_prev = t_cam
            continue
        dt = min(max(t_cam - t_prev, 1e-3), 0.5)
        t_prev = t_cam
        xi, P, info = ekfm.ekf(xi, P, f_I, dt, params, p_C_payload=p_C,
                               phi=phi, theta=theta, psi=psi, C_CM=C_CM)
        rows.append((t_cam, t, len(det), dof(C_CM), info['nis'],
                     xi[0], xi[1], xi[2], xi[3], xi[4],
                     math.sqrt(P[0, 0]), math.sqrt(P[1, 1]), phi, theta, psi))
    return pd.DataFrame(rows, columns=['t_cam', 't_flight', 'n', 'dof', 'nis',
                                       'alpha_x', 'alpha_y',
                                       'alpha_dot_x', 'alpha_dot_y', 'psi_p',
                                       'sigma_alpha_x', 'sigma_alpha_y',
                                       'phi', 'theta', 'psi'])


def mean_nnis(df, warmup=3.0):
    if len(df) == 0:
        return np.inf
    m = df[(df.n > 0) & (df.t_cam > df.t_cam.min() + warmup)]
    if len(m) < 20:
        return np.inf
    return float(np.mean(m.nis.to_numpy()/m.dof.to_numpy()))


def make_params(L, **kw):
    kw.setdefault("q_alpha", 0.02**2)
    kw.setdefault("q_psi_p", 0.3**2)
    kw.setdefault("sigma_bearing", math.radians(0.05))
    kw.setdefault("sigma_yaw", math.radians(3.0))
    kw.setdefault("sigma_att", math.radians(0.5))
    return ekfm.EKFParams(L=L, **kw)


if __name__ == "__main__":
    import argparse
    import time

    ap = argparse.ArgumentParser(
        description="Sweep (clock offset, tether length) and report the pair "
                    "that minimises NIS. Put the winner in sessions.toml.")
    ap.add_argument("selector", nargs="?", default="latest.cam",
                    help="which session (see `uv run plot.py ls`)")
    args = ap.parse_args()

    session = catalog.resolve(args.selector)
    if not session.has_camera:
        raise SystemExit(f"session {session.id} has no camera data")
    print(f"{session.id}  {session.label}")
    print(f"   currently sessions.toml says pose_offset = "
          f"{session.pose_offset:+.2f} s")

    inp = build_inputs(session.fl)
    frames = group_frames(session.poses)

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
    print(f"\nBEST: offset {offs[j]:+.2f} s, L_eff {Ls[i]:.2f} m, "
          f"NIS {table[i, j]:.2f}")
    print(f"\nrecord it:\n  [sessions.{session.id}]\n"
          f"  pose_offset = {offs[j]:.2f}")
