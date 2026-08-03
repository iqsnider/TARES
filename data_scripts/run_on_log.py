"""Run the payload EKF against a real flight log + pose CSV."""
import numpy as np
import pandas as pd
import math
import sim.estimation.ekf as ekfm
import sim.estimation.pre_process as pp
import catalog

G = 9.80665


def build_inputs(fl):
    """Inertial-ENU specific force a_I + attitude vs flight time. Computed once."""
    a_frd = fl[['drone_ax_meas', 'drone_ay_meas', 'drone_az_meas']].to_numpy()*G / \
        1000.
    phi = fl.drone_roll.to_numpy()
    theta = fl.drone_pitch.to_numpy()
    psi = fl.drone_yaw.to_numpy()
    a_I = np.empty_like(a_frd)
    for i in range(len(fl)):
        a_I[i] = ekfm.T_IB_fn(phi[i], theta[i], psi[i]
                              ) @ (ekfm.P_SWAP @ a_frd[i])
    return dict(t=fl.cur_time.to_numpy(), a_I=a_I, phi=phi, theta=theta, psi=psi)


def group_frames(pose):
    """[(t_cam, frame, det), ...] sorted by time.

    The filter pre-processes the camera data itself now, so it gets the frame
    whole; `det` is that frame's detected rows, kept for the marker count and
    for the plots.
    """
    out = []
    for _, g in pose.groupby('frame'):
        out.append((float(g.time_s.iloc[0]), g, g.dropna(subset=['marker_id'])))
    out.sort(key=lambda kv: kv[0])
    return out


def make_ekf(phi, theta, psi, alpha_x, alpha_y, psi_p, L=None, **kw):
    """Build the filter, optionally overriding the tether length.

    L is not a constructor argument -- EKF reads config.TETHER_LEN -- so the
    sweep sets the attribute afterwards.
    """
    filt = ekfm.EKF(phi, theta, psi, alpha_x, alpha_y, psi_p, **kw)
    if L is not None:
        filt.L = L
    return filt


def run(frames, inp, offset, L=None):
    t_fl, a_fl = inp['t'], inp['a_I']
    filt = None
    t_prev = None
    rows = []
    for t_cam, g, det in frames:
        t = t_cam - offset
        if t < t_fl[0] or t > t_fl[-1]:
            continue
        phi = np.interp(t, t_fl, inp['phi'])
        theta = np.interp(t, t_fl, inp['theta'])
        psi = np.interp(t, t_fl, inp['psi'])
        a_I = np.array([np.interp(t, t_fl, a_fl[:, 0]),
                        np.interp(t, t_fl, a_fl[:, 1]),
                        np.interp(t, t_fl, a_fl[:, 2])])
        if filt is None:
            seed = pp.swing_angles(g, ekfm.T_IB_fn(phi, theta, psi))
            if seed is None:
                continue
            filt = make_ekf(phi, theta, psi, *seed, L=L)
            t_prev = t_cam
            continue
        dt = min(max(t_cam - t_prev, 1e-3), 0.5)
        t_prev = t_cam
        # nis is only written when a frame actually updates the filter
        filt.nis = None
        xi, P = filt(g, a_I, dt, phi, theta, psi)
        rows.append((t_cam, t, len(det), filt.nis,
                     xi[0], xi[1], xi[2], xi[3], xi[4],
                     math.sqrt(P[0, 0]), math.sqrt(P[1, 1]), phi, theta, psi))
    return pd.DataFrame(rows, columns=['t_cam', 't_flight', 'n', 'nis',
                                       'alpha_x', 'alpha_y',
                                       'alpha_dot_x', 'alpha_dot_y', 'psi_p',
                                       'sigma_alpha_x', 'sigma_alpha_y',
                                       'phi', 'theta', 'psi'])


def mean_nnis(df, warmup=3.0):
    """NIS per measurement row. Every update folds in all MEAS_DIM rows."""
    if len(df) == 0:
        return np.inf
    m = df[(df.n > 0) & (df.t_cam > df.t_cam.min() + warmup)]
    if len(m) < 20:
        return np.inf
    return float(np.mean(m.nis.to_numpy()/ekfm.MEAS_DIM))


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
        for j, o in enumerate(offs):
            table[i, j] = mean_nnis(run(frames, inp, o, L=L))
    print(f"({time.time()-t0:.0f}s)\n{'L_eff':>7} | best offset    min NIS")
    for i, L in enumerate(Ls):
        j = int(np.argmin(table[i]))
        print(f"{L:>7.2f} | {offs[j]:>+10.2f} s {table[i, j]:>10.2f}")
    i, j = np.unravel_index(np.argmin(table), table.shape)
    print(f"\nBEST: offset {offs[j]:+.2f} s, L_eff {Ls[i]:.2f} m, "
          f"NIS {table[i, j]:.2f}")
    print(f"\nrecord it:\n  [sessions.{session.id}]\n"
          f"  pose_offset = {offs[j]:.2f}")
