import math
import numpy as np
import cv2

import sim.config as config
import estimation.calculate_payload_position as payload


# NED ENU swap
P_SWAP = np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, -1]])

STATE_DIM = 5
IX_AX, IX_AY, IX_DAX, IX_DAY, IX_YAWP = range(STATE_DIM)


def to_inertial_frame_from_body_frame(phi, theta, psi):
    """
    Transformation matrix to the inertial frame from the body fixed frame
    """
    sp, cp = math.sin(phi), math.cos(phi)
    st, ct = math.sin(theta), math.cos(theta)
    sy, cy = math.sin(psi), math.cos(psi)

    return np.array([[ct*cy, sp*st*cy - cp*sy, cp*st*cy + sp*sy],
                     [ct*sy, sp*st*sy + cp*cy, cp*st*sy - sp*cy],
                     [-st, sp*ct, cp*ct]])


def C_nb_enu(roll, pitch, yaw):
    """
    Camera rotation from (starboard, nose, up) body -> ENU inertial
    """
    return P_SWAP @ to_inertial_frame_from_body_frame(roll, pitch, yaw) @ P_SWAP


def n_hat(alpha_x, alpha_y):
    """
    Unit vector from pivot toward the payload, INERTIAL frame
    """
    sx, cx = math.sin(alpha_x), math.cos(alpha_x)
    sy, cy = math.sin(alpha_y), math.cos(alpha_y)

    return np.array([sx*cy,
                     sy,
                     -cx*cy])


def dn_hat(alpha_x, alpha_y):
    """
    d n_hat / d(alpha_x, alpha_y), 3X2 matrix
    """
    sx, cx = math.sin(alpha_x), math.cos(alpha_x)
    sy, cy = math.sin(alpha_y), math.cos(alpha_y)

    return np.array([[cx*cy, -sx*sy],
                     [0, cy],
                     [sx*cy, cx*sy]])


def m_hat(yaw_p):
    """
    Unit vector along the marker row, INERTIAL frame (horizontal).
    """
    return np.array([math.cos(yaw_p),
                     math.sin(yaw_p),
                     0])


def dm_hat(yaw_p):
    """
    d m_hat / d yaw_p, 3X1 matrix
    """
    return np.array([-math.sin(yaw_p),
                     math.cos(yaw_p),
                     0])


def angles_from_direction(n_inertial):
    """
    Returns (alpha_x, alpha_y) from any pivot->payload vector
    expressed in the inertial frame. Used primarily for initialization
    """
    n = np.asarray(n_inertial, dtype=float)
    n = n / np.linalg.norm(n)
    alpha_y = math.asin(np.clip(n[1], -1, 1))
    alpha_x = math.atan2(n[0], -n[2])

    return alpha_x, alpha_y


class EKFParams:
    """
    Everything the filter needs that isn't state, input, or measurement
    """

    def __init__(self,
                 mtx,  # 3x3 camera matrix
                 dist,  # distortion coefficients
                 L=None,  # tether length for DYNAMICS [m]
                 L_m=None,  # pivot -> marker board center [m]
                 C_bc=None,  # camera -> body
                 t_bc=None,  # camera optical center in body [m]
                 l_piv=None,  # tether pivot in body frame [m]
                 q=(0.02)**2,  # process noise on ddalpha [rad^2/s^3]
                 q_yaw=(0.3)**2,  # process noise on yaw_p [rad^2/s]
                 sigma_det=0.4,  # marker centroid noise [px]
                 sigma_att=math.radians(0.5)):  # attitude 1-sigma [rad]

        self.mtx = np.asarray(mtx, dtype=float)
        self.dist = np.asarray(dist, dtype=float)
        self.L = config.TETHER_LEN if L is None else L
        self.L_m = self.L if L_m is None else L_m
        self.C_bc = config.CAM_R if C_bc is None else np.asarray(C_bc, float)
        self.C_cb = self.C_bc.T
        self.t_bc = (np.array([config.CAM_OFFSET_X,
                               config.CAM_OFFSET_Y,
                               config.CAM_OFFSET_Z])
                     if t_bc is None else np.asarray(t_bc, float))
        self.l_piv = np.zeros(3) if l_piv is None else np.asarray(l_piv, float)
        self.q = q
        self.q_yaw = q_yaw
        self.sigma_det = sigma_det
        self.sigma_att = sigma_att

    @property
    def f_u(self):
        return self.mtx[0, 0]

    @property
    def f_v(self):
        return self.mtx[1, 1]

    def pixel_variance(self):
        """
        Detection noise plus attitude error mapped into pixels
        """
        return self.sigma_det**2 + (self.f_u * self.sigma_att)**2


def f_dynamics(xi, u, L):
    """
    nonlinear pendulum model
    u is specific force in the INERTIAL frame.
    [dax, day, ddax, dday, dyaw_p]
    payload yaw rate is purely a placeholder for a later algebraic computation
    """
    ax, ay, dax, day, _ = xi
    u1, u2, u3 = u
    sx, cx = math.sin(ax), math.cos(ax)
    sy, cy = math.sin(ay), math.cos(ay)

    ddax = -(cx*u1 + sx*u3)/(L*cy) + 2*(sy/cy)*dax*day
    dday = -(cy*u2 + sy*(cx*u3 - sx*u1))/L - sy*cy*dax**2

    return np.array([dax,
                     day,
                     ddax,
                     dday,
                     0])


def F_jacobian(xi, u, L):
    """
    F 5X5 matrix
    """
    ax, ay, dax, day, _ = xi
    u1, u2, u3 = u
    sx, cx = math.sin(ax), math.cos(ax)
    sy, cy = math.sin(ay), math.cos(ay)

    F = np.zeros((STATE_DIM, STATE_DIM))
    F[IX_AX, IX_DAX] = 1
    F[IX_AY, IX_DAY] = 1

    F[IX_DAX, IX_AX] = (sx*u1 - cx*u3)/(L*cy)
    F[IX_DAX, IX_AY] = -(cx*u1 + sx*u3)*sy/(L*cy**2) + 2*dax*day/cy**2
    F[IX_DAX, IX_DAX] = 2*(sy/cy)*day
    F[IX_DAX, IX_DAY] = 2*(sy/cy)*dax

    F[IX_DAY, IX_AX] = sy*(cx*u1 + sx*u3)/L
    F[IX_DAY, IX_AY] = (sy*u2 - cy*(cx*u3 - sx*u1))/L - math.cos(2*ay)*dax**2
    F[IX_DAY, IX_DAX] = -2*sy*cy*dax

    # yaw_p row stays zero: random walk
    return F


def rk4_step(xi, u, dt, L):
    """
    RK4 Integration
    """
    k1 = f_dynamics(xi, u, L)
    k2 = f_dynamics(xi + 0.5*dt*k1, u, L)
    k3 = f_dynamics(xi + 0.5*dt*k2, u, L)
    k4 = f_dynamics(xi + dt*k3, u, L)

    return xi + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)


def marker_prediction(xi, offset, C_bn, params):
    """
    Predicted pixel and Jacobian for one marker.

    xi: payload state
    offset: MARKER_OFFSET[marker_id], signed, along the marker row [m]
    C_bn: inertial -> body rotation
    params: EKFParams
    """
    ax, ay, _, _, yaw_p = xi
    n = n_hat(ax, ay)
    m = m_hat(yaw_p)

    # marker position relative to the pivot, inertial frame
    p_n = params.L_m*n - offset*m

    # expressed in body, then relative to the camera, then in camera frame
    p_b = params.l_piv + C_bn @ p_n - params.t_bc
    p_c = params.C_cb @ p_b

    X, Y, Z = p_c

    f_u, f_v = params.f_u, params.f_v
    u0, v0 = params.mtx[0, 2], params.mtx[1, 2]
    h = np.array([f_u*X/Z + u0, f_v*Y/Z + v0])

    # d(pixel)/d(p_c)
    dpi = np.array([[f_u/Z, 0, -f_u*X/Z**2],
                    [0, f_v/Z, -f_v*Y/Z**2]])
    # d(pixel)/d(p_n)
    M = dpi @ params.C_cb @ C_bn

    H = np.zeros((2, STATE_DIM))
    H[:, IX_AX:IX_AY+1] = params.L_m*(M @ dn_hat(ax, ay))
    H[:, IX_YAWP] = -offset*(M @ dm_hat(yaw_p))

    return h, H, p_c


def undistort_pixels(uv, params):
    """
    Raw image pixels -> ideal pinhole pixels. uv NX2 matrix
    """
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.undistortPoints(uv, params.mtx, params.dist, P=params.mtx)
    return out.reshape(-1, 2)


def ekf_predict(xi, P, u, dt, params):
    """
    Propagate state and covariance one step. u is inertial specific force
    """
    xi = np.asarray(xi, dtype=float)
    P = np.asarray(P, dtype=float)

    xi_pred = rk4_step(xi, u, dt, params.L)

    F = F_jacobian(xi, u, params.L)
    Fdt = F*dt
    Phi = np.eye(STATE_DIM) + Fdt + 0.5*(Fdt @ Fdt)

    Q = np.diag([0, 0, params.q, params.q, params.q_yaw]) * dt
    P_pred = Phi @ P @ Phi.T + Q
    P_pred = 0.5*(P_pred + P_pred.T)

    return xi_pred, P_pred


def ekf_update(xi, P, measurements, roll, pitch, yaw, params, already_undistorted=False):
    """
    Fold in every marker detected in one camera frame

    measurements  iterable of (marker_id, u_px, v_px), raw image pixels
    roll/pitch/yaw  drone attitude at the IMAGE timestamp, ArduPilot NED
                    convention (the flight log's drone_roll/pitch/yaw)
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

    C_bn = C_nb_enu(roll, pitch, yaw).T

    z_rows, h_rows, H_rows = [], [], []
    for marker_id, z_uv in zip(ids, uv):
        offset = payload.MARKER_OFFSET[marker_id]
        h, H, _ = marker_prediction(xi, offset, C_bn, params)
        if h is None:
            continue
        z_rows.append(z_uv)
        h_rows.append(h)
        H_rows.append(H)

    if not z_rows:
        return xi, P, info

    z = np.concatenate(z_rows)
    h = np.concatenate(h_rows)
    H = np.vstack(H_rows)
    R = params.pixel_variance() * np.eye(z.size)

    nu = z - h
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    xi = xi + K @ nu
    xi[IX_YAWP] = math.atan2(math.sin(xi[IX_YAWP]), math.cos(xi[IX_YAWP]))

    P = (np.eye(STATE_DIM) - K @ H) @ P
    P = 0.5*(P + P.T)

    info["n_markers"] = len(z_rows)
    info["innovation"] = nu
    info["nis"] = float(nu @ np.linalg.solve(S, nu))

    return xi, P, info


def ekf(xi, P, u, dt, params, measurements=None, roll=0, pitch=0, yaw=0, already_undistorted=False):
    """
    One full filter tick: predict, then update if a camera frame arrived

    u: 3X1 drone specific force, INERTIAL frame. Hover = [0, 0, +g]
    """
    xi, P = ekf_predict(xi, P, u, dt, params)
    if measurements:
        xi, P, info = ekf_update(xi, P, measurements, roll, pitch, yaw,
                                 params, already_undistorted)
    else:
        info = {"n_markers": 0, "innovation": None, "nis": None}

    return xi, P, info


def initial_state(payload_center_camera_frame, roll, pitch, yaw, marker_R=None, sigma_alpha=0.02):
    """
    Seed the filter from one frame's PnP output. Initialization only
    """
    C_nb = C_nb_enu(roll, pitch, yaw)
    p_b = config.CAM_R @ np.asarray(payload_center_camera_frame, float)
    p_n = C_nb @ p_b
    alpha_x, alpha_y = angles_from_direction(p_n)

    if marker_R is not None:
        m_n = C_nb @ (config.CAM_R @ np.asarray(marker_R, float)[:, 0])
        yaw_p = math.atan2(m_n[1], m_n[0])
        sigma_yaw = math.radians(15)
    else:
        yaw_p = 0
        sigma_yaw = math.pi

    xi = np.array([alpha_x, alpha_y, 0, 0, yaw_p])
    P = np.diag([sigma_alpha**2, sigma_alpha**2, 1, 1, sigma_yaw**2])

    return xi, P
