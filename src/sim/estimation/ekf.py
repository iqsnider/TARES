import math
import numpy as np
import cv2

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


def n_I(alpha_x, alpha_y):
    """
    nI: unit vector from pivot toward the payload, INERTIAL frame
    """
    sax, cax = math.sin(alpha_x), math.cos(alpha_x)
    say, cay = math.sin(alpha_y), math.cos(alpha_y)

    return np.array([sax*cay,
                     say,
                     -cax*cay])


def dn_I_dalpha(alpha_x, alpha_y):
    """
    d nI / d(alpha_x, alpha_y), 3X2 matrix
    """
    sax, cax = math.sin(alpha_x), math.cos(alpha_x)
    say, cay = math.sin(alpha_y), math.cos(alpha_y)

    return np.array([[cax*cay, -sax*say],
                     [0, cay],
                     [sax*cay, cax*say]])


def m_I(psi_p):
    """
    mI: unit vector along the marker row, INERTIAL frame (horizontal)
    """
    return np.array([math.cos(psi_p),
                     math.sin(psi_p),
                     0])


def dm_I_dpsi_p(psi_p):
    """
    d mI / d psi_p, 3X1 matrix
    """
    return np.array([-math.sin(psi_p),
                     math.cos(psi_p),
                     0])


def alpha_from_n_I(n):
    """
    Returns (alpha_x, alpha_y) from any pivot->payload vector
    expressed in the inertial frame. Used primarily for initialization
    """
    n = np.asarray(n, dtype=float)
    n = n / np.linalg.norm(n)
    alpha_y = math.asin(np.clip(n[1], -1, 1))
    alpha_x = math.atan2(n[0], -n[2])

    return alpha_x, alpha_y


class EKFParams:
    """
    Everything the filter needs that isn't state, input, or measurement
    """

    def __init__(self,
                 mtx, # 3x3 camera matrix
                 dist, # distortion coefficients
                 L=None, # tether length for DYNAMICS [m]
                 L_m=None, # pivot -> marker board center [m]
                 C_BC=None, # camera -> body
                 t_BC_B=None, # camera optical center in body [m]
                 l_B=None, # tether pivot in body frame [m]
                 q_alpha=(0.02)**2, # process noise on alpha_ddot
                 q_psi_p=(0.3)**2, # process noise on psi_p
                 sigma_det=0.4, # marker centroid noise [px]
                 sigma_att=math.radians(0.5)): # attitude 1-sigma [rad]

        self.mtx = np.asarray(mtx, dtype=float)
        self.dist = np.asarray(dist, dtype=float)
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
        self.sigma_det = sigma_det
        self.sigma_att = sigma_att

    @property
    def f_u(self):
        return self.mtx[0, 0]

    @property
    def f_v(self):
        return self.mtx[1, 1]

    @property
    def u_0(self):
        return self.mtx[0, 2]

    @property
    def v_0(self):
        return self.mtx[1, 2]

    def pixel_variance(self):
        """
        Detection noise plus attitude error mapped into pixels
        """
        return self.sigma_det**2 + (self.f_u*self.sigma_att)**2


def xi_dot_I(xi, f_I, L):
    """
    [alpha_dot_x, alpha_dot_y, alpha_ddot_x, alpha_ddot_y, 0]

    f_I is specific force in the inertial frame, so m_D is already folded in
    """
    alpha_x, alpha_y, alpha_dot_x, alpha_dot_y, _ = xi
    fx, fy, fz = f_I
    sax, cax = math.sin(alpha_x), math.cos(alpha_x)
    say, cay = math.sin(alpha_y), math.cos(alpha_y)

    alpha_ddot_x = (-(cax*fx + sax*fz)/(L*cay)
                    + 2*(say/cay)*alpha_dot_x*alpha_dot_y)
    alpha_ddot_y = (-(cay*fy + say*(cax*fz - sax*fx))/L
                    - say*cay*alpha_dot_x**2)

    return np.array([alpha_dot_x,
                     alpha_dot_y,
                     alpha_ddot_x,
                     alpha_ddot_y,
                     0])


def F_jacobian(xi, f_I, L):
    """
    F = jacobian(xi_dot_I, xi_I), 5X5 matrix
    """
    alpha_x, alpha_y, alpha_dot_x, alpha_dot_y, _ = xi
    fx, fy, fz = f_I
    sax, cax = math.sin(alpha_x), math.cos(alpha_x)
    say, cay = math.sin(alpha_y), math.cos(alpha_y)

    F = np.zeros((STATE_DIM, STATE_DIM))
    F[IX_ALPHA_X, IX_ALPHA_DOT_X] = 1
    F[IX_ALPHA_Y, IX_ALPHA_DOT_Y] = 1

    F[IX_ALPHA_DOT_X, IX_ALPHA_X] = (sax*fx - cax*fz)/(L*cay)
    F[IX_ALPHA_DOT_X, IX_ALPHA_Y] = (-(cax*fx + sax*fz)*say/(L*cay**2)
                                     + 2*alpha_dot_x*alpha_dot_y/cay**2)
    F[IX_ALPHA_DOT_X, IX_ALPHA_DOT_X] = 2*(say/cay)*alpha_dot_y
    F[IX_ALPHA_DOT_X, IX_ALPHA_DOT_Y] = 2*(say/cay)*alpha_dot_x

    F[IX_ALPHA_DOT_Y, IX_ALPHA_X] = say*(cax*fx + sax*fz)/L
    F[IX_ALPHA_DOT_Y, IX_ALPHA_Y] = ((say*fy - cay*(cax*fz - sax*fx))/L
                                     - math.cos(2*alpha_y)*alpha_dot_x**2)
    F[IX_ALPHA_DOT_Y, IX_ALPHA_DOT_X] = -2*say*cay*alpha_dot_x
    return F


def rk4_step(xi, f_I, dt, L):
    """
    RK4 Integration
    """
    k1 = xi_dot_I(xi, f_I, L)
    k2 = xi_dot_I(xi + 0.5*dt*k1, f_I, L)
    k3 = xi_dot_I(xi + 0.5*dt*k2, f_I, L)
    k4 = xi_dot_I(xi + dt*k3, f_I, L)

    return xi + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


def marker_prediction(xi, o_j, C_BI, params):
    """
    Predicted pixel h_j and Jacobian H_j for one marker.

    xi: payload state xi_I
    o_j: MARKER_OFFSET[marker_id], signed, along the marker row [m]
    C_BI: rotation to body from inertial
    params: EKFParams
    """
    alpha_x, alpha_y, _, _, psi_p = xi
    n = n_I(alpha_x, alpha_y)
    m = m_I(psi_p)

    # marker position relative to the pivot, inertial frame
    p_I_j = params.L_m*n - o_j*m

    # expressed in body, then relative to the camera, then in camera frame
    p_B_j = params.l_B + C_BI @ p_I_j - params.t_BC_B
    p_C_j = params.C_CB @ p_B_j

    X_j, Y_j, Z_j = p_C_j

    f_u, f_v = params.f_u, params.f_v
    u_0, v_0 = params.u_0, params.v_0
    h_j = np.array([f_u*X_j/Z_j + u_0, f_v*Y_j/Z_j + v_0])

    # d h_j / d p_C_j
    dh_dpC = np.array([[f_u/Z_j, 0, -f_u*X_j/Z_j**2],
                       [0, f_v/Z_j, -f_v*Y_j/Z_j**2]])
    # d h_j / d p_I_j
    dh_dpI = dh_dpC @ params.C_CB @ C_BI

    # H_j = dh_dpC * C_CB * C_BI * dpI_dxi
    H_j = np.zeros((2, STATE_DIM))
    H_j[:, IX_ALPHA_X:IX_ALPHA_Y+1] = params.L_m*(dh_dpI @ dn_I_dalpha(alpha_x, alpha_y))
    H_j[:, IX_PSI_P] = -o_j*(dh_dpI @ dm_I_dpsi_p(psi_p))

    return h_j, H_j, p_C_j


def undistort_pixels(uv, params):
    """
    Raw image pixels -> ideal pinhole pixels. uv NX2 matrix
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(uv, params.mtx, params.dist, P=params.mtx)
    return out.reshape(-1, 2)


def ekf_predict(xi, P, f_I, dt, params):
    """
    Propagate state and covariance one step. f_I is inertial specific force
    """
    xi = np.asarray(xi, dtype=float)
    P = np.asarray(P, dtype=float)

    xi_pred = rk4_step(xi, f_I, dt, params.L)

    F = F_jacobian(xi, f_I, params.L)
    Fdt = F*dt
    Phi = np.eye(STATE_DIM) + Fdt + 0.5*(Fdt @ Fdt)

    Q = np.diag([0, 0, params.q_alpha, params.q_alpha, params.q_psi_p])*dt
    P_pred = Phi @ P @ Phi.T + Q
    P_pred = 0.5*(P_pred + P_pred.T)

    return xi_pred, P_pred


def ekf_update(xi, P, measurements, phi, theta, psi, params, already_undistorted=False):
    """
    Fold in every marker detected in one camera frame
    """
    xi = np.asarray(xi, dtype=float).copy()
    P = np.asarray(P, dtype=float).copy()
    info = {"n_markers": 0, "innovation": None, "nis": None}

    meas = [m for m in (measurements or [])
            if not (np.isnan(m[1]) or np.isnan(m[2]))]
    if not meas:
        return xi, P, info

    ids = [int(m[0]) for m in meas]
    uv = np.array([[m[1], m[2]] for m in meas], dtype=float)
    if not already_undistorted:
        uv = undistort_pixels(uv, params)

    C_BI = C_IB_enu(phi, theta, psi).T

    z_rows, h_rows, H_rows = [], [], []
    for marker_id, z_uv in zip(ids, uv):
        o_j = payload.MARKER_OFFSET[marker_id]
        h_j, H_j, _ = marker_prediction(xi, o_j, C_BI, params)
        if h_j is None:
            continue
        z_rows.append(z_uv)
        h_rows.append(h_j)
        H_rows.append(H_j)

    if not z_rows:
        return xi, P, info

    z = np.concatenate(z_rows)
    h = np.concatenate(h_rows)
    H = np.vstack(H_rows)
    R = params.pixel_variance() * np.eye(z.size)

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


def ekf(xi, P, f_I, dt, params, measurements=None, phi=0, theta=0, psi=0, already_undistorted=False):
    """
    One full filter tick: predict, then update if a camera frame arrived

    f_I: 3X1 drone specific force, INERTIAL frame. Hover = [0, 0, +g0]
    """
    xi, P = ekf_predict(xi, P, f_I, dt, params)
    if measurements:
        xi, P, info = ekf_update(xi, P, measurements, phi, theta, psi,
                                 params, already_undistorted)
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
    alpha_x, alpha_y = alpha_from_n_I(p_I)

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
