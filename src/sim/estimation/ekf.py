"""
EKF for tracking a swinging payload.

Pre-processing does all the camera work: PnP on the three ArUco markers, the
board center, and the marker-row orientation. What reaches the filter is the
payload center in the camera frame and (optionally) the marker rotation.

    z = [b_x, b_y, psi_P]

    b       unit line of sight from the tether pivot to the payload,
            camera frame. Only the first two components are used, the
            third is redundant since ||b|| = 1
    psi_P   payload yaw, inertial frame

Small-angle model throughout, with n = 1:

    [q]^I = [alpha_x, alpha_y, -1],   A = C_CB C_BI

    h(xi) = [ (A [q]^I)_x, (A [q]^I)_y, psi_P ]

so h is LINEAR in xi and H is constant within a timestep. Bearing bias from
the truncation is about alpha^3/6, under a milliradian below 10 deg of swing.

Camera frame is the OpenCV convention config.CAM_R assumes:
+x right, +y DOWN, +z along the optical axis.
"""
import math
import numpy as np

import sim.config as config


# NED <-> ENU swap
P_SWAP = np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, -1]])

# xi_I = [alpha_x, alpha_y, alpha_dot_x, alpha_dot_y, psi_p]
STATE_DIM = 5
IX_ALPHA_X, IX_ALPHA_Y, IX_ALPHA_DOT_X, IX_ALPHA_DOT_Y, IX_PSI_P = range(STATE_DIM)

# z = [b_x, b_y, psi_p]
MEAS_DIM = 3
IX_B_X, IX_B_Y, IX_PSI_MEAS = range(MEAS_DIM)


def wrap_pi(a):
    """
    Wrap an angle to (-pi, pi]
    """
    return math.atan2(math.sin(a), math.cos(a))


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
    [q]^I: small-angle vector from pivot toward the payload, INERTIAL frame.

    Not normalised: ||[q]^I|| = sqrt(1 + alpha_x^2 + alpha_y^2), taken as 1
    """
    return np.array([alpha_x, alpha_y, -1.0])


def alpha_from_q_I(q):
    """
    Returns (alpha_x, alpha_y) from any pivot->payload vector expressed in
    the inertial frame. Used primarily for initialization
    """
    q = np.asarray(q, dtype=float)
    q = q/np.linalg.norm(q)
    alpha_y = math.asin(np.clip(q[1], -1, 1))
    alpha_x = math.atan2(q[0], -q[2])

    return alpha_x, alpha_y


class EKFParams:
    """
    Everything the filter needs that isn't state, input, or measurement
    """

    def __init__(self,
                 L=None, # tether pivot -> payload CG [m]
                 C_BC=None, # camera -> body
                 t_BC_B=None, # camera optical center in body [m]
                 l_B=None, # tether pivot in body frame [m]
                 q_alpha=(0.02)**2, # process noise on alpha_ddot
                 q_psi_p=(0.3)**2, # process noise on psi_p
                 sigma_bearing=math.radians(0.05), # bearing noise [rad]
                 sigma_yaw=math.radians(3.0), # payload yaw noise [rad]
                 sigma_att=math.radians(0.5)): # drone attitude 1-sigma [rad]

        self.L = config.TETHER_LEN if L is None else L
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
        self.sigma_yaw = sigma_yaw
        self.sigma_att = sigma_att

    def bearing_variance(self):
        """
        Bearing noise plus drone attitude error [rad^2]
        """
        return self.sigma_bearing**2 + self.sigma_att**2

    def yaw_variance(self):
        """
        Payload yaw noise plus drone attitude error [rad^2]
        """
        return self.sigma_yaw**2 + self.sigma_att**2


def bearing_from_p_C(p_C_payload, params):
    """
    Payload center in the camera frame -> unit line of sight FROM THE TETHER
    PIVOT, still in the camera frame.

    Shifts the origin camera -> body -> pivot, then normalises
    """
    p_B = params.C_BC @ np.asarray(p_C_payload, float) + params.t_BC_B
    p_C = params.C_CB @ (p_B - params.l_B)

    return p_C/np.linalg.norm(p_C)


def yaw_from_C_CM(C_CM, C_IB, params):
    """
    Payload yaw from the PnP marker rotation. First column of C_CM is the
    marker row direction in the camera frame
    """
    m_I = C_IB @ (params.C_BC @ np.asarray(C_CM, float)[:, 0])

    return math.atan2(m_I[1], m_I[0])


def xi_dot_I(xi, f_I, L):
    """
    Small-angle process model

    [alpha_dot_x,
     alpha_dot_y,
     -(a_1 + alpha_x |g|)/L,
     -(a_2 + alpha_y |g|)/L,
     0]

    f_I is specific force in the inertial frame, so m_D is already folded in
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


def measurement_prediction(xi, C_BI, params):
    """
    Predicted measurement h and Jacobian H for one camera frame.

    h = [q^C_x, q^C_y, psi_p], H is 3X5 and constant within the timestep
    """
    alpha_x, alpha_y, _, _, psi_p = xi
    A = params.C_CB @ C_BI
    q_C = A @ q_I(alpha_x, alpha_y)

    h = np.array([q_C[0], q_C[1], psi_p])

    H = np.zeros((MEAS_DIM, STATE_DIM))
    H[0:2, IX_ALPHA_X:IX_ALPHA_Y+1] = A[0:2, 0:2]
    H[IX_PSI_MEAS, IX_PSI_P] = 1

    return h, H


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


def ekf_update(xi, P, p_C_payload, phi, theta, psi, params, C_CM=None):
    """
    Fold in one camera frame.

    p_C_payload: payload board center in the camera frame [m], from PnP
    C_CM: marker -> camera rotation from PnP. Omit it and the yaw row is
          dropped, leaving psi_p to coast on the process model
    """
    xi = np.asarray(xi, dtype=float).copy()
    P = np.asarray(P, dtype=float).copy()
    info = {"innovation": None, "nis": None}

    if p_C_payload is None or np.isnan(p_C_payload).any():
        return xi, P, info

    C_IB = C_IB_enu(phi, theta, psi)
    C_BI = C_IB.T

    b = bearing_from_p_C(p_C_payload, params)
    h, H = measurement_prediction(xi, C_BI, params)

    z = np.array([b[0], b[1], 0.0])
    R = np.diag([params.bearing_variance(),
                 params.bearing_variance(),
                 params.yaw_variance()])
    rows = 2

    if C_CM is not None:
        z[IX_PSI_MEAS] = yaw_from_C_CM(C_CM, C_IB, params)
        rows = MEAS_DIM

    z, h, H, R = z[:rows], h[:rows], H[:rows], R[:rows, :rows]

    y = z - h
    if rows == MEAS_DIM:
        y[IX_PSI_MEAS] = wrap_pi(y[IX_PSI_MEAS])

    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    xi = xi + K @ y
    xi[IX_PSI_P] = wrap_pi(xi[IX_PSI_P])

    P = (np.eye(STATE_DIM) - K @ H) @ P
    P = 0.5*(P + P.T)

    info["innovation"] = y
    info["nis"] = float(y @ np.linalg.solve(S, y))

    return xi, P, info


def ekf(xi, P, f_I, dt, params, p_C_payload=None, phi=0, theta=0, psi=0, C_CM=None):
    """
    One full filter tick: predict, then update if a camera frame arrived

    f_I: 3X1 drone specific force, INERTIAL frame. Hover = [0, 0, +g0]
    """
    xi, P = ekf_predict(xi, P, f_I, dt, params)
    if p_C_payload is not None:
        xi, P, info = ekf_update(xi, P, p_C_payload, phi, theta, psi, params, C_CM)
    else:
        info = {"innovation": None, "nis": None}

    return xi, P, info


def initial_state(p_C_payload, phi, theta, psi, params, C_CM=None, sigma_alpha=0.02):
    """
    Seed the filter from one frame's PnP output
    """
    C_IB = C_IB_enu(phi, theta, psi)
    b_C = bearing_from_p_C(p_C_payload, params)
    alpha_x, alpha_y = alpha_from_q_I(C_IB @ (params.C_BC @ b_C))

    if C_CM is not None:
        psi_p = yaw_from_C_CM(C_CM, C_IB, params)
        sigma_psi_p = math.radians(15)
    else:
        psi_p = 0
        sigma_psi_p = math.pi

    xi = np.array([alpha_x, alpha_y, 0, 0, psi_p])
    P = np.diag([sigma_alpha**2, sigma_alpha**2, 1, 1, sigma_psi_p**2])

    return xi, P
