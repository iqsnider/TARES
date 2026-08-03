"""
EKF for tracking a swinging payload from ArUco marker bearings.

Pre-processing hands the filter unit line-of-sight vectors in the camera
frame, so nothing here touches pixels, intrinsics, or distortion. The
geometry is otherwise unchanged: each marker is predicted individually,
including L_m, the pivot and camera lever arms, and the marker offset o_j.

    z_j = [b_x, b_y] for marker j, unit bearing in the camera frame

    h_j = first two components of [p_j]^C / ||[p_j]^C||

Camera frame is the OpenCV convention config.CAM_R assumes:
+x right, +y DOWN, +z along the optical axis.
"""
import math
import numpy as np

import sim.config as config
import sim.estimation.calculate_payload_position as payload


# NED <-> ENU swap
P_SWAP = np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, -1]])

# xi_I = [alpha_x, alpha_y, alpha_dot_x, alpha_dot_y, psi_p]
STATE_DIM = 5
IX_ALPHA_X, IX_ALPHA_Y, IX_ALPHA_DOT_X, IX_ALPHA_DOT_Y, IX_PSI_P = range(STATE_DIM)


def C_IB_from_euler(phi, theta, psi):
    """
    C_IB: rotation to the inertial frame from the body fixed frame
    """
    sp, cp = math.sin(phi), math.cos(phi)
    st, ct = math.sin(theta), math.cos(theta)
    sy, cy = math.sin(psi), math.cos(psi)

    return np.array([[ct*cy, sp*st*cy - cp*sy, cp*st*cy + sp*sy],
                     [ct*sy, sp*st*sy + cp*cy, cp*st*sy - sp*cy],
                     [-st, sp*ct, cp*ct]])


def C_IB_enu(phi, theta, psi):
    """
    C_IB for ENU body (starboard, nose, up) -> ENU inertial
    """
    return P_SWAP @ C_IB_from_euler(phi, theta, psi) @ P_SWAP


def q_I(alpha_x, alpha_y):
    """
    [q]^I: UNIT vector from pivot toward the payload, INERTIAL frame
    """
    return np.array([alpha_x, alpha_y, -1])


def dq_I_dalpha(alpha_x, alpha_y):
    """
    d [q]^I / d(alpha_x, alpha_y)
    """
    return np.array([[1,0],[0,1],[0,0]])


def m_I(psi_p):
    """
    [m]^I: unit vector along the marker row, INERTIAL frame (horizontal)
    """
    return np.array([math.cos(psi_p),
                     math.sin(psi_p),
                     0])


def dm_I_dpsi_p(psi_p):
    """
    d [m]^I / d psi_p, 3X1 matrix
    """
    return np.array([-math.sin(psi_p),
                     math.cos(psi_p),
                     0])


def alpha_from_q_I(q):
    """
    Returns (alpha_x, alpha_y) from any pivot->payload vector
    expressed in the inertial frame. Used primarily for initialization
    """
    q = np.asarray(q, dtype=float)
    q = q / np.linalg.norm(q)
    alpha_y = math.asin(np.clip(q[1], -1, 1))
    alpha_x = math.atan2(q[0], -q[2])

    return alpha_x, alpha_y


class EKFParams:
    """
    Everything the filter needs that isn't state, input, or measurement
    """

    def __init__(self,
                 L=None, # tether length for DYNAMICS [m]
                 L_m=None, # pivot -> marker board center [m]
                 C_BC=None, # camera -> body
                 t_BC_B=None, # camera optical center in body [m]
                 l_B=None, # tether pivot in body frame [m]
                 q_alpha=(0.02)**2, # process noise on alpha_ddot
                 q_psi_p=(0.3)**2, # process noise on psi_p
                 sigma_bearing=math.radians(0.04), # marker bearing noise [rad]
                 sigma_att=math.radians(0.5)): # attitude 1-sigma [rad]

        self.L = config.TETHER_LEN if L is None else L
        self.L_m = self.L if L_m is None else L_m
        self.C_BC = config.CAM_R if C_BC is None else np.asarray(C_BC, float)
        self.C_CB = self.C_BC.T
        self.t_BC_B = (np.array([config.CAM_OFFSET_X,
                                 config.CAM_OFFSET_Y,
                                 config.CAM_OFFSET_Z])
                       if t_BC_B is None else np.asarray(t_BC_B, float))
        self.l_B = np.zeros(3) if l_B is None else np.asarray(l_B, float)
        self.q_alpha = q_alpha
        self.q_psi_p = q_psi_p
        self.sigma_bearing = sigma_bearing
        self.sigma_att = sigma_att

    def bearing_variance(self):
        """
        Detection noise plus attitude error, as an angle [rad^2]
        """
        return self.sigma_bearing**2 + self.sigma_att**2


def xi_dot_I(xi, f_I, L):
    """
    process model
    f_I is specific force in the inertial frame
    """
    alpha_x, alpha_y, alpha_dot_x, alpha_dot_y, _ = xi
    g = config.GRAVITY

    return np.array([alpha_dot_x,
                     alpha_dot_y,
                     -(f_I[0] + alpha_x*g)/L,
                     -(f_I[1] + alpha_y*g)/L,
                     0])


def F_jacobian(xi, f_I, L):
    """
    F = jacobian(xi_dot_I, xi_I), 5X5 matrix.

    Constant under the small-angle model. xi and f_I are unused, kept so the
    signature matches the nonlinear version
    """
    g = config.GRAVITY

    F = np.zeros((STATE_DIM, STATE_DIM))
    F[IX_ALPHA_X, IX_ALPHA_DOT_X] = 1
    F[IX_ALPHA_Y, IX_ALPHA_DOT_Y] = 1
    F[IX_ALPHA_DOT_X, IX_ALPHA_X] = -g/L
    F[IX_ALPHA_DOT_Y, IX_ALPHA_Y] = -g/L

    return F


def euler_step(xi, f_I, dt, L):
    """
    Euler integration
    """
    return xi + dt*xi_dot_I(xi, f_I, L)


def marker_prediction(xi, o_j, C_BI, params):
    """
    Predicted bearing h_j and Jacobian H_j for one marker.

    xi: payload state xi_I
    o_j: MARKER_OFFSET[marker_id], signed, along the marker row [m]
    C_BI: rotation to body from inertial
    params: EKFParams
    """
    alpha_x, alpha_y, _, _, psi_p = xi
    q = q_I(alpha_x, alpha_y)
    m = m_I(psi_p)

    # marker position relative to the pivot, inertial frame
    p_I_j = params.L_m*q - o_j*m

    # expressed in body, then relative to the camera, then in camera frame
    p_B_j = params.l_B + C_BI @ p_I_j - params.t_BC_B
    p_C_j = params.C_CB @ p_B_j

    X_j, Y_j, Z_j = p_C_j
    rho = math.sqrt(X_j**2 + Y_j**2 + Z_j**2)

    h_j = np.array([X_j/rho, Y_j/rho])

    # d h_j / d p_C_j
    dh_dpC = np.array([[rho**2 - X_j**2, -X_j*Y_j, -X_j*Z_j],
                       [-X_j*Y_j, rho**2 - Y_j**2, -Y_j*Z_j]])/rho**3
    # d h_j / d p_I_j
    dh_dpI = dh_dpC @ params.C_CB @ C_BI

    # H_j = dh_dpC * C_CB * C_BI * dpI_dxi
    H_j = np.zeros((2, STATE_DIM))
    H_j[:, IX_ALPHA_X:IX_ALPHA_Y+1] = params.L_m*(dh_dpI @ dq_I_dalpha(alpha_x, alpha_y))
    H_j[:, IX_PSI_P] = -o_j*(dh_dpI @ dm_I_dpsi_p(psi_p))

    return h_j, H_j, p_C_j


def ekf_predict(xi, P, f_I, dt, params):
    """
    Propagate state and covariance one step. f_I is inertial specific force
    """
    xi = np.asarray(xi, dtype=float)
    P = np.asarray(P, dtype=float)

    xi_pred = euler_step(xi, f_I, dt, params.L)

    F = F_jacobian(xi, f_I, params.L)
    Phi = np.eye(STATE_DIM) + F*dt

    Q = np.diag([0, 0, params.q_alpha, params.q_alpha, params.q_psi_p])*dt
    P_pred = Phi @ P @ Phi.T + Q
    P_pred = 0.5*(P_pred + P_pred.T)

    return xi_pred, P_pred


def ekf_update(xi, P, measurements, phi, theta, psi, params):
    """
    Fold in every marker detected in one camera frame

    measurements: list of (marker_id, b_x, b_y, b_z), unit line of sight in
    the camera frame
    """
    xi = np.asarray(xi, dtype=float).copy()
    P = np.asarray(P, dtype=float).copy()
    info = {"n_markers": 0, "innovation": None, "nis": None}

    meas = [m for m in (measurements or []) if not np.isnan(m[1:4]).any()]
    if not meas:
        return xi, P, info

    ids = [int(m[0]) for m in meas]
    b = np.array([m[1:4] for m in meas], dtype=float)
    b = b/np.linalg.norm(b, axis=1, keepdims=True)

    C_BI = C_IB_enu(phi, theta, psi).T

    z_rows, h_rows, H_rows = [], [], []
    for marker_id, b_j in zip(ids, b):
        o_j = payload.MARKER_OFFSET[marker_id]
        h_j, H_j, _ = marker_prediction(xi, o_j, C_BI, params)
        if h_j is None:
            continue
        z_rows.append(b_j[0:2])
        h_rows.append(h_j)
        H_rows.append(H_j)

    if not z_rows:
        return xi, P, info

    z = np.concatenate(z_rows)
    h = np.concatenate(h_rows)
    H = np.vstack(H_rows)
    R = params.bearing_variance() * np.eye(z.size)

    y = z - h
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    xi = xi + K @ y
    xi[IX_PSI_P] = math.atan2(math.sin(xi[IX_PSI_P]), math.cos(xi[IX_PSI_P]))

    P = (np.eye(STATE_DIM) - K @ H) @ P
    P = 0.5*(P + P.T)

    info["n_markers"] = len(z_rows)
    info["innovation"] = y
    info["nis"] = float(y @ np.linalg.solve(S, y))

    return xi, P, info


def ekf(xi, P, f_I, dt, params, measurements=None, phi=0, theta=0, psi=0):
    """
    One full filter tick: predict, then update if a camera frame arrived

    f_I: 3X1 drone specific force, INERTIAL frame. Hover = [0, 0, +g0]
    """
    xi, P = ekf_predict(xi, P, f_I, dt, params)
    if measurements:
        xi, P, info = ekf_update(xi, P, measurements, phi, theta, psi, params)
    else:
        info = {"n_markers": 0, "innovation": None, "nis": None}

    return xi, P, info


def initial_state(p_C_payload, phi, theta, psi, C_CM=None, sigma_alpha=0.02):
    """
    Seed the filter from one frame's PnP output.

    p_C_payload: payload board center in the camera frame
    C_CM: marker -> camera rotation from PnP, first column is mI in camera frame
    """
    C_IB = C_IB_enu(phi, theta, psi)
    p_B = config.CAM_R @ np.asarray(p_C_payload, float)
    p_I = C_IB @ p_B
    alpha_x, alpha_y = alpha_from_q_I(p_I)

    if C_CM is not None:
        m = C_IB @ (config.CAM_R @ np.asarray(C_CM, float)[:, 0])
        psi_p = math.atan2(m[1], m[0])
        sigma_psi_p = math.radians(15)
    else:
        psi_p = 0
        sigma_psi_p = math.pi

    xi = np.array([alpha_x, alpha_y, 0, 0, psi_p])
    P = np.diag([sigma_alpha**2, sigma_alpha**2, 1, 1, sigma_psi_p**2])

    return xi, P
