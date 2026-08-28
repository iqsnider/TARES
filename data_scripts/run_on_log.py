import numpy as np
import math
import sim.estimation.ekf as ekfm
import sim.estimation.pre_process as pp
import sim.transformations as tf

G = 9.80665


def build_inputs(fl):
    """
    Drone acceleration and attitude from flight log
    """
    a_frd = fl[['drone_ax_meas', 'drone_ay_meas', 'drone_az_meas']].to_numpy()*G / 1000
    phi = fl.drone_roll.to_numpy()
    theta = fl.drone_pitch.to_numpy()
    psi = fl.drone_yaw.to_numpy()
    a_I = np.empty_like(a_frd)

    for i in range(len(fl)):
        a_ned = tf.T_IB(phi[i], theta[i], psi[i]) @ a_frd[i]
        a_I[i] = tf.T_ENU_from_NED() @ a_ned

    return dict(t=fl.cur_time.to_numpy(), a_I=a_I, phi=phi, theta=theta, psi=psi)


def group_frames(pose):
    """
    Groups marker ids together with a timestamp for all frames in the pose file
    """
    out = []
    for _, g in pose.groupby('frame'):
        out.append((float(g.time_s.iloc[0]), g, g.dropna(subset=['marker_id'])))

    return out


def make_ekf(phi, theta, psi, alpha_x, alpha_y, psi_p, **kwargs):
    """
    Build the EKF
    """
    # filt = ekfm.EKF(phi, theta, psi, alpha_x, alpha_y, psi_p, **kwargs)
    filt = ekfm.EKF(phi, theta, psi, alpha_x, alpha_y, psi_p, **kwargs)

    return filt


def run_full(frames, inp, offset, geom=None, hold=None, **ekf_kwargs):
    """
    Run the EKF over every camera frame.

    `geom` defaults to pre_process.DEFAULT_GEOMETRY (today's Prm/config.py);
    pass a session's own instead, from its config_snapshot.json. Filter tuning
    in `ekf_kwargs` goes the same way, straight to the EKF.

    `hold(t_flight)` marks the stretches where the run cut the camera on
    purpose. Those frames are fed to the filter as no detection at all, so a
    replay exercises the prediction over the same blackouts the test did.
    """
    geom = pp.DEFAULT_GEOMETRY if geom is None else geom
    t_fl = inp["t"]
    S = tf.T_ENU_from_NED()
    filt = None
    t_prev = None
    out = []
    for t_cam, frame, det in frames:
        t = t_cam - offset

        phi = np.interp(t, t_fl, inp["phi"])
        theta = np.interp(t, t_fl, inp["theta"])
        psi = np.interp(t, t_fl, inp["psi"])
        a_I = np.array([np.interp(t, t_fl, inp["a_I"][:, k]) for k in range(3)])

        # a held frame reaches the filter as nothing, the way an occluded one does
        held = hold is not None and hold(t)
        seen = None if held else frame

        if filt is None:
            if held:
                continue
            T_IB = S @ tf.T_IB(phi, theta, psi) @ S
            init_ekf = pp.swing_angles(frame, T_IB, geom=geom)

            if init_ekf is None:
                continue

            filt = make_ekf(phi, theta, psi, *init_ekf, geom=geom,
                            **ekf_kwargs)
            t_prev = t_cam

            continue

        dt = t_cam - t_prev
        t_prev = t_cam
        xi, P = filt(seen, a_I, dt, phi, theta, psi)

        out.append(dict(t_cam=t_cam, t_flight=t, n=0 if held else len(det),
                        frame=int(frame.frame.iloc[0]),
                        xi=xi.copy(), P=P.copy(), T_IB=filt.T_IB.copy(),
                        alpha_x=xi[ekfm.IX_ALPHA_X],
                        alpha_y=xi[ekfm.IX_ALPHA_Y],
                        alpha_dot_x=xi[ekfm.IX_ALPHA_DOT_X],
                        alpha_dot_y=xi[ekfm.IX_ALPHA_DOT_Y],
                        psi_p=xi[ekfm.IX_PSI_P],
                        sigma_alpha_x=math.sqrt(P[ekfm.IX_ALPHA_X,
                                                 ekfm.IX_ALPHA_X]),
                        sigma_alpha_y=math.sqrt(P[ekfm.IX_ALPHA_Y,
                                                 ekfm.IX_ALPHA_Y]),
                        sigma_psi_p=math.sqrt(P[ekfm.IX_PSI_P, ekfm.IX_PSI_P]),
                        q_I=filt.q_I(xi[ekfm.IX_ALPHA_X],
                                     xi[ekfm.IX_ALPHA_Y])))
    return out
