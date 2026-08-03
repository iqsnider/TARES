import math
import numpy as np

import Prm.config as config
import sim.estimation.pre_process as pp


# NED <-> ENU swap
P_SWAP = np.array([[0, 1, 0],
                   [1, 0, 0],
                   [0, 0, -1]])

# xi_I = [alpha_x, alpha_y, alpha_dot_x, alpha_dot_y, psi_p]
STATE_DIM = 5
IX_ALPHA_X, IX_ALPHA_Y, IX_ALPHA_DOT_X, IX_ALPHA_DOT_Y, IX_PSI_P = range(STATE_DIM)

# z = [b_x, b_y, z_psi_p], built by pre_process
MEAS_DIM = pp.MEAS_DIM
IX_B_X, IX_B_Y, IX_PSI_MEAS = pp.IX_B_X, pp.IX_B_Y, pp.IX_PSI_MEAS


def wrap_pi(a):
    """
    Wrap an angle to (-pi, pi]
    """
    return math.atan2(math.sin(a), math.cos(a))


def T_IB_fn(phi, theta, psi):
    """
    Rotation to the inertial frame from the body frame.

    Body is ENU-ordered (starboard, nose, up) and inertial is ENU, so the
    standard NED euler matrix is sandwiched between the swap
    """
    sp, cp = math.sin(phi), math.cos(phi)
    st, ct = math.sin(theta), math.cos(theta)
    sy, cy = math.sin(psi), math.cos(psi)

    T_ned = np.array([[ct*cy, sp*st*cy - cp*sy, cp*st*cy + sp*sy],
                      [ct*sy, sp*st*sy + cp*cy, cp*st*sy - sp*cy],
                      [-st, sp*ct, cp*ct]])

    return P_SWAP @ T_ned @ P_SWAP


class EKF:
    """
    Initializes an EKF for the uav swing payload system
    Calling the EKF will give the estimate
    """

    def __init__(self,
                 initial_phi, initial_theta, initial_psi, # iniital drone attitude
                 initial_alpha_x, initial_alpha_y, initial_psi_p, # initial payload swing states
                 q_xy=(0.02)**2, # process noise on alpha_ddot_xy
                 q_yaw=(0.3)**2, # process noise on psi_p
                 sigma_xy=math.radians(0.5), # bearing noise [rad]
                 sigma_yaw=math.radians(3), # payload yaw noise [rad]
                 sigma_alpha_0=math.radians(2), # initial swing angle 1-sigma [rad]
                 sigma_rate_0=math.radians(30), # initial swing rate 1-sigma [rad/s]
                 sigma_psi_p_0=math.radians(15)): # initial payload yaw 1-sigma [rad]

        # access constant geometry parrameters from the config. The camera
        # lever arms live in pre_process, which is what needs them
        self.L = config.TETHER_LEN
        self.T_BC = config.CAM_R
        self.T_CB = self.T_BC.T
        self.g = config.GRAVITY

        # set noise parameters
        self.q_xy = q_xy
        self.q_yaw = q_yaw
        self.sigma_xy = sigma_xy
        self.sigma_yaw = sigma_yaw

        # initialize EKF
        self.T_IB = T_IB_fn(initial_phi, initial_theta, initial_psi)
        self.xi = np.array([initial_alpha_x, initial_alpha_y, 0, 0, initial_psi_p])
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
        alpha_x, alpha_y, alpha_dot_x, alpha_dot_y, _ = xi

        xi_dot_I = np.array([alpha_dot_x,
                             alpha_dot_y,
                             -(a_I[0] + alpha_x*g)/L,
                             -(a_I[1] + alpha_y*g)/L,
                             0])

        return xi_dot_I

    def F_jacobian(self):
        """
        Jacobian of the process model
        """
        g = self.g
        L = self.L

        F = np.zeros((STATE_DIM, STATE_DIM))
        F[IX_ALPHA_X, IX_ALPHA_DOT_X] = 1
        F[IX_ALPHA_Y, IX_ALPHA_DOT_Y] = 1
        F[IX_ALPHA_DOT_X, IX_ALPHA_X] = -g/L
        F[IX_ALPHA_DOT_Y, IX_ALPHA_Y] = -g/L

        return F

    def euler_integration(self, xi, a_I, dt):
        """
        Euler integration
        """
        return xi + dt*self.process_model(xi, a_I)

    def ekf_predict(self, xi, P, a_I, dt):
        """
        Propagate state and covariance one step
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

        P_pred = 0.5*(P_pred + P_pred.T)

        return xi_pred, P_pred

    def measurement_prediction(self, xi, T_BI):
        """
        Predicted measurement h and Jacobian H for one camera frame.
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

        return h, H

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
        xi = np.asarray(xi, dtype=float).copy()
        P = np.asarray(P, dtype=float).copy()

        # all the camera work: PnP -> board center -> line of sight and yaw
        z = pp.measurement(frame, T_IB)
        if z is None:
            return xi, P

        T_BI = T_IB.T
        h, H = self.measurement_prediction(xi, T_BI)

        # innovation, yaw wrapped
        y = z - h
        y[IX_PSI_MEAS] = wrap_pi(y[IX_PSI_MEAS])

        # measurement noise
        R = np.diag([self.sigma_xy**2,
                     self.sigma_xy**2,
                     self.sigma_yaw**2])

        # innovation covariance
        S = H @ P @ H.T + R

        # kalman gain
        K = P @ H.T @ np.linalg.inv(S)
        xi = xi + K @ y
        xi[IX_PSI_P] = wrap_pi(xi[IX_PSI_P])
        P = (np.eye(STATE_DIM) - K @ H) @ P
        P = 0.5*(P + P.T)

        self.nis = float(y @ np.linalg.solve(S, y))

        return xi, P

    def __call__(self, frame, a_I, dt, phi, theta, psi):
        """
        Performs full EKF tick
        """
        self.T_IB = T_IB_fn(phi, theta, psi)

        self.xi, self.P = self.ekf_predict(self.xi, self.P, a_I, dt)
        if frame is not None:
            self.xi, self.P = self.ekf_update(self.xi, self.P, frame, self.T_IB)

        return self.xi, self.P
