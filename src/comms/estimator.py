"""
EKF lifecycle helpers for the swinging-payload estimator.
"""

import numpy as np

import sim.estimation.ekf as ekf_lib
import sim.estimation.pre_process as pp


# full control state is 16 long; the payload swing states live at the tail
IX_ALPHA_X_16, IX_ALPHA_Y_16 = 12, 13
IX_ALPHA_DOT_X_16, IX_ALPHA_DOT_Y_16 = 14, 15


def start_ekf(logger, recorder=None, alpha_x=0, alpha_y=0):
    """
    Build an EKF seeded from the current drone attitude.
    """
    c = logger.cache
    phi, theta, psi = c["roll"], c["pitch"], c["yaw"]

    psi_p = psi                                  # fallback: assume aligned
    if recorder is not None:
        _, poses = recorder.latest_poses()
        z = pp.measurement_from_poses(poses, ekf_lib.T_IB_fn(phi, theta, psi))
        if z is not None:
            psi_p = z[pp.IX_PSI_MEAS]

    return ekf_lib.EKF(phi, theta, psi, alpha_x, alpha_y, psi_p)



def step_ekf(ekf, poses, a_I, dt, phi, theta, psi):
    """
    One predict, plus one update if a new camera frame is available
    """
    ekf.T_IB = ekf_lib.T_IB_fn(phi, theta, psi)

    xi, P = ekf.ekf_predict(ekf.xi, ekf.P, a_I, dt)

    if poses:
        z = pp.measurement_from_poses(poses, ekf.T_IB)
        if z is not None:
            xi, P = ekf.update_with_z(xi, P, z, ekf.T_IB)

    ekf.xi, ekf.P = xi, P

    return xi, P


def payload_state_16(x_drone, xi):
    """
    Assemble the 16 state vector OuterLoopPayloadLQR expects.
    """
    x16 = np.zeros(16)
    x16[0:6] = x_drone
    x16[IX_ALPHA_X_16] = xi[ekf_lib.IX_ALPHA_X]
    x16[IX_ALPHA_Y_16] = xi[ekf_lib.IX_ALPHA_Y]
    x16[IX_ALPHA_DOT_X_16] = xi[ekf_lib.IX_ALPHA_DOT_X]
    x16[IX_ALPHA_DOT_Y_16] = xi[ekf_lib.IX_ALPHA_DOT_Y]
    return x16
