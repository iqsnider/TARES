import math
import numpy as np
import cv2

import Prm.config as config
import sim.estimation.pre_process as pp
import sim.transformations as tf

# xi_I = [alpha_x, alpha_y, alpha_dot_x, alpha_dot_y, psi_p]
STATE_DIM = 5
IX_ALPHA_X, IX_ALPHA_Y, IX_ALPHA_DOT_X, IX_ALPHA_DOT_Y, IX_PSI_P = range(
    STATE_DIM)

# z = [b_x, b_y, z_psi_p], built by pre_process
MEAS_DIM = pp.MEAS_DIM
IX_B_X, IX_B_Y, IX_PSI_MEAS = pp.IX_B_X, pp.IX_B_Y, pp.IX_PSI_MEAS

# which tracker feeds the filter. A marker board carries an orientation, so it
# measures payload yaw as well as bearing; a circle is symmetric and does not
SOURCE_ARUCO = "aruco"
SOURCE_COLOR = "color"
MEAS_DIM_BY_SOURCE = {SOURCE_ARUCO: 3, SOURCE_COLOR: 2}


def wrap_pi(a):
    """
    Wrap an angle to (-pi, pi]
    """
    return math.atan2(math.sin(a), math.cos(a))


class EKF:
    """
    Initializes an EKF for the uav swing payload system
    Calling the EKF will give the estimate
    """

    def __init__(self,
                 initial_phi, initial_theta, initial_psi,  # iniital drone attitude
                 initial_alpha_x, initial_alpha_y, initial_psi_p,  # initial payload swing states
                 q_xy=None,  # process noise on alpha_ddot_xy
                 q_yaw=None,  # process noise on psi_p
                 sigma_xy=None,  # bearing noise [rad]
                 sigma_yaw=None,  # payload yaw noise [rad]
                 sigma_alpha_0=None,  # initial swing angle 1-sigma [rad]
                 sigma_rate_0=None,  # initial swing rate 1-sigma [rad/s]
                 sigma_psi_p_0=None,  # initial payload yaw 1-sigma [rad]
                 zeta=None,  # swing damping ratio
                 source=None,  # aruco or color
                 L=None, g=None, geom=None):

        # default to live Prm/config.py; pass these explicitly to replay a
        # session with its own config_snapshot.json values instead
        self.L = config.TETHER_LEN if L is None else L
        self.g = config.GRAVITY if g is None else g
        self.zeta = config.EKF_ZETA if zeta is None else zeta
        self.geom = pp.DEFAULT_GEOMETRY if geom is None else geom
        self.T_BC = self.geom.T_BC
        self.T_CB = self.geom.T_CB
        self.t_BC_B = self.geom.t_BC_B
        self.l_B = self.geom.l_B

        sigma_alpha_0 = (config.EKF_SIGMA_ALPHA_0 if sigma_alpha_0 is None
                         else sigma_alpha_0)
        sigma_rate_0 = (config.EKF_SIGMA_RATE_0 if sigma_rate_0 is None
                        else sigma_rate_0)
        sigma_psi_p_0 = (config.EKF_SIGMA_PSI_P_0 if sigma_psi_p_0 is None
                         else sigma_psi_p_0)

        # set noise parameters
        self.q_xy = config.EKF_Q_XY if q_xy is None else q_xy
        self.q_yaw = config.EKF_Q_YAW if q_yaw is None else q_yaw
        self.sigma_xy = config.EKF_SIGMA_XY if sigma_xy is None else sigma_xy
        self.sigma_yaw = config.EKF_SIGMA_YAW if sigma_yaw is None else sigma_yaw

        self.source = config.EKF_SOURCE if source is None else source
        self.meas_dim = MEAS_DIM_BY_SOURCE[self.source]

        # last measured minus predicted, None until a measurement lands
        self.innov = None

        # initialize EKF
        S = tf.T_ENU_from_NED()
        self.T_IB = S @ tf.T_IB(initial_phi, initial_theta, initial_psi) @ S
        self.xi = np.array(
            [initial_alpha_x, initial_alpha_y, 0, 0, initial_psi_p])
        self.P = np.diag([sigma_alpha_0**2,
                          sigma_alpha_0**2,
                          sigma_rate_0**2,
                          sigma_rate_0**2,
                          sigma_psi_p_0**2])

    def process_model(self, xi, a_I):
        """
        5 state process model for the EKF
        """
        g = self.g
        L = self.L
        z = self.zeta
        w = math.sqrt(g/L)
        alpha_x, alpha_y, alpha_dot_x, alpha_dot_y, _ = xi

        xi_dot_I = np.array([alpha_dot_x,
                             alpha_dot_y,
                             -(a_I[0] + alpha_x*g)/L - 2*z*w*alpha_dot_x,
                             -(a_I[1] + alpha_y*g)/L - 2*z*w*alpha_dot_y,
                             0])

        return xi_dot_I

    def F_jacobian(self):
        """
        Jacobian of the process model
        """
        g = self.g
        L = self.L
        w = math.sqrt(g/L)

        F = np.zeros((STATE_DIM, STATE_DIM))
        F[IX_ALPHA_X, IX_ALPHA_DOT_X] = 1
        F[IX_ALPHA_Y, IX_ALPHA_DOT_Y] = 1
        F[IX_ALPHA_DOT_X, IX_ALPHA_X] = -g/L
        F[IX_ALPHA_DOT_Y, IX_ALPHA_Y] = -g/L
        F[IX_ALPHA_DOT_X, IX_ALPHA_DOT_X] = -2*self.zeta*w
        F[IX_ALPHA_DOT_Y, IX_ALPHA_DOT_Y] = -2*self.zeta*w

        return F

    def euler_integration(self, xi, a_I, dt):
        """
        Improved Euler (Heun's method) integration.

        a_I is held constant across the step, since only one acceleration
        sample is available per predict call.
        """
        k1 = self.process_model(xi, a_I)
        k2 = self.process_model(xi + dt*k1, a_I)

        return xi + dt/2*(k1 + k2)

    def ekf_predict(self, xi, P, a_I, dt):
        """
        Propagate state and covariance one step.
        """
        xi = np.asarray(xi, dtype=float)
        P = np.asarray(P, dtype=float)

        xi_pred = self.euler_integration(xi, a_I, dt)

        # process jacobian
        F = self.F_jacobian()

        # discrete time process jacobian
        Phi = np.eye(STATE_DIM) + F*dt

        Q = np.diag([0, 0, self.q_xy, self.q_xy, self.q_yaw])*dt
        P_pred = Phi @ P @ Phi.T + Q

        return xi_pred, P_pred

    def measurement_prediction(self, xi, T_BI):
        """
        Predicted measurement h and Jacobian H for one camera frame. Both are
        cut to the tracker's width, so a color source drops the yaw row it
        cannot measure.
        """
        alpha_x, alpha_y, _, _, psi_p = xi
        A = self.T_CB @ T_BI
        q_C = A @ self.q_I(alpha_x, alpha_y)

        # measurement prediction from swing angles
        h = np.array([q_C[0], q_C[1], psi_p])

        # Jacobian
        H = np.zeros((MEAS_DIM, STATE_DIM))
        H[0:2, IX_ALPHA_X:IX_ALPHA_Y+1] = A[0:2, 0:2]
        H[IX_PSI_MEAS, IX_PSI_P] = 1

        rows = self.meas_dim

        return h[:rows], H[:rows]

    @staticmethod
    def q_I(alpha_x, alpha_y):
        """
        Approximate unit vector from drone to payload
        """
        return np.array([alpha_x, alpha_y, -1])

    def ekf_update(self, xi, P, frame, T_IB):
        """
        Fold in one camera frame
        """
        z = pp.measurement(frame, T_IB, geom=self.geom)
        if z is None:
            return xi, P

        return self.update_with_z(xi, P, z, T_IB)

    def update_with_z(self, xi, P, z, T_IB):
        """
        Fold in one measurement vector
        """
        xi = np.asarray(xi, dtype=float).copy()
        P = np.asarray(P, dtype=float).copy()

        T_BI = T_IB.T
        h, H = self.measurement_prediction(xi, T_BI)

        # innovation, yaw wrapped when the tracker measures one
        y = z - h
        if self.meas_dim > IX_PSI_MEAS:
            y[IX_PSI_MEAS] = wrap_pi(y[IX_PSI_MEAS])

        # measured minus predicted, kept so a caller can log what the camera
        # disagreed with the model about
        self.innov = y.copy()

        # measurement noise
        R = np.diag([self.sigma_xy**2,
                     self.sigma_xy**2,
                     self.sigma_yaw**2][:self.meas_dim])

        # innovation covariance
        S = H @ P @ H.T + R

        # kalman gain
        K = P @ H.T @ np.linalg.inv(S)
        xi = xi + K @ y
        xi[IX_PSI_P] = wrap_pi(xi[IX_PSI_P])
        P = (np.eye(STATE_DIM) - K @ H) @ P

        return xi, P

    def estimate_to_px_coords(self, xi, P, T_IB, K, D, n_sigma=1):
        """
        Takes a payload state estimate and calculates the pixel coordinates sized to the camera frame.
        For viewing the estimate overlaid on the recording.
        """
        L = self.L
        A = self.T_CB @ T_IB.T

        q_I = np.array([xi[IX_ALPHA_X], xi[IX_ALPHA_Y], -1])

        # undo pivot shift
        p_C = L*(A @ q_I) - self.T_CB @ (self.t_BC_B - self.l_B)

        uv, _ = cv2.projectPoints(p_C.reshape(
            1, 1, 3), np.zeros(3), np.zeros(3), K, D)
        center = uv.ravel()

        X, Y, Z = p_C
        J_proj = np.array([[K[0, 0]/Z, 0, -K[0, 0]*X/Z**2],
                           [0, K[1, 1]/Z, -K[1, 1]*Y/Z**2]])
        J = J_proj @ (L*A[:, 0:2])
        P_px = J @ P[0:2, 0:2] @ J.T

        evals, evecs = np.linalg.eigh(P_px)
        axes = n_sigma*np.sqrt(evals[::-1])      # major first
        angle = math.degrees(math.atan2(evecs[1, -1], evecs[0, -1]))

        return center, axes, angle

    def estimate_to_swing_velocity(self, xi):
        """
        Payload velocity along inertial east and north
        """
        L = self.L
        alpha_x, alpha_y, alpha_dot_x, alpha_dot_y, _ = xi

        # sax, cax = math.sin(alpha_x), math.cos(alpha_x)
        # say, cay = math.sin(alpha_y), math.cos(alpha_y)

        # v_x = L*(cax*cay*alpha_dot_x - sax*say*alpha_dot_y)
        # v_y = L*cay*alpha_dot_y
        v_x = L*alpha_dot_x
        v_y = L*alpha_dot_y

        return v_x, v_y

    def __call__(self, frame, a_I, dt, phi, theta, psi):
        """
        Performs full EKF tick
        """
        S = tf.T_ENU_from_NED()
        self.T_IB = S @ tf.T_IB(phi, theta, psi) @ S

        self.xi, self.P = self.ekf_predict(self.xi, self.P, a_I, dt)
        if frame is not None:
            self.xi, self.P = self.ekf_update(
                self.xi, self.P, frame, self.T_IB)

        return self.xi, self.P
