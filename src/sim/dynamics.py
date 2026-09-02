import math
import numpy as np
from scipy.linalg import solve_continuous_are
import Prm.config as config

HOVER_THRUST = config.MASS_TOTAL*config.GRAVITY


def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def tether_equilibrium_state(p_pl_ref, v_pl_ref, tether_length):
    """
    Build x* in R^16 from the payload reference
    """
    x_ref = np.zeros(16)
    x_ref[0:3] = p_pl_ref + np.array([0, 0, tether_length])  # drone pos
    x_ref[3:6] = v_pl_ref  # drone vel
    # x_ref[6:9]  drone attitude = 0
    # x_ref[9:12] drone ang vel = 0
    # x_ref[12:14] tether angles  = 0
    # x_ref[14:16] tether rates = 0
    return x_ref


def _build_double_integrator():
    """
    Linear translational model of the drone
    state: [px py pz vx vy vz] input: [ax ay az]
    """
    A = np.zeros((6, 6))
    A[0, 3] = 1   # px_dot = vx
    A[1, 4] = 1   # py_dot = vy
    A[2, 5] = 1   # pz_dot = vz

    B = np.zeros((6, 3))
    B[3, 0] = 1   # vx_dot = ax
    B[4, 1] = 1   # vy_dot = ay
    B[5, 2] = 1   # vz_dot = az
    return A, B


def _lqr(A, B, Q, R):
    """Continuous-time LQR gain: u = -K x."""
    P = solve_continuous_are(A, B, Q, R)
    return np.linalg.inv(R) @ B.T @ P


class OuterLoopLQR:
    """
    6 state lqr for just drone
    """

    def __init__(self,
                 q_pos_xy=None, q_pos_z=None,
                 q_vel_xy=None, q_vel_z=None,
                 r_acc=None):
        q_pos_xy = config.LQR_DRONE_Q_POS_XY if q_pos_xy is None else q_pos_xy
        q_pos_z = config.LQR_DRONE_Q_POS_Z if q_pos_z is None else q_pos_z
        q_vel_xy = config.LQR_DRONE_Q_VEL_XY if q_vel_xy is None else q_vel_xy
        q_vel_z = config.LQR_DRONE_Q_VEL_Z if q_vel_z is None else q_vel_z
        r_acc = config.LQR_DRONE_R_ACC if r_acc is None else r_acc
        A, B = _build_double_integrator()

        # state cost
        Q = np.diag([q_pos_xy, q_pos_xy, q_pos_z,
                     q_vel_xy, q_vel_xy, q_vel_z])
        # control cost
        R = r_acc * np.eye(3)

        self.K = _lqr(A, B, Q, R)  # (3, 6)

    def compute_u(self, x, p_ref, v_ref, a_ref=None):
        p_ref = np.asarray(p_ref)
        v_ref = np.asarray(v_ref)
        if a_ref is None:
            a_ref = np.zeros(3)

        e = np.concatenate([x[0:3] - p_ref, x[3:6] - v_ref])

        return a_ref - self.K @ e


class OuterLoopLQI:
    """
    6 state lqi for just drone, 3x6 state gain + 3x3 integral gain
    """

    def __init__(self,
                 q_pos_xy=None, q_pos_z=None,
                 q_vel_xy=None, q_vel_z=None,
                 q_int_xy=None, q_int_z=None,
                 r_acc=None, e_band=None, u_i_max=None):
        q_pos_xy = config.LQI_DRONE_Q_POS_XY if q_pos_xy is None else q_pos_xy
        q_pos_z = config.LQI_DRONE_Q_POS_Z if q_pos_z is None else q_pos_z
        q_vel_xy = config.LQI_DRONE_Q_VEL_XY if q_vel_xy is None else q_vel_xy
        q_vel_z = config.LQI_DRONE_Q_VEL_Z if q_vel_z is None else q_vel_z
        q_int_xy = config.LQI_DRONE_Q_INT_XY if q_int_xy is None else q_int_xy
        q_int_z = config.LQI_DRONE_Q_INT_Z if q_int_z is None else q_int_z
        r_acc = config.LQI_DRONE_R_ACC if r_acc is None else r_acc
        e_band = config.LQI_DRONE_E_BAND if e_band is None else e_band
        u_i_max = config.LQI_DRONE_U_I_MAX if u_i_max is None else u_i_max

        A, B = _build_double_integrator()

        # the output the integrator accumulates: drone position
        C = np.zeros((3, 6))
        C[0, 0] = 1
        C[1, 1] = 1
        C[2, 2] = 1

        # augment: xI_dot = C @ x
        Abar = np.zeros((9, 9))
        Abar[:6, :6] = A
        Abar[6:, :6] = C
        Bbar = np.zeros((9, 3))
        Bbar[:6, :] = B

        Q = np.zeros((9, 9))
        Q[:6, :6] = np.diag([q_pos_xy, q_pos_xy, q_pos_z,
                             q_vel_xy, q_vel_xy, q_vel_z])
        Q[6:, 6:] = np.diag([q_int_xy, q_int_xy, q_int_z])
        R = r_acc*np.eye(3)

        Kbar = _lqr(Abar, Bbar, Q, R)   # (3, 9)
        self.C = C
        self.K = Kbar[:, :6]            # (3, 6)
        self.Ki = Kbar[:, 6:]           # (3, 3)
        self.xi = np.zeros(3)

        # make accessible for analysis
        self.Abar = Abar
        self.Bbar = Bbar
        self.Kbar = Kbar
        self.Q = Q
        self.R = R

        self.e_band = e_band
        self.u_i_max = u_i_max

    def compute_u(self, x, p_ref, v_ref, a_ref=None):
        dt = 1/config.CONTROL_FREQUENCY
        p_ref = np.asarray(p_ref)
        v_ref = np.asarray(v_ref)
        if a_ref is None:
            a_ref = np.zeros(3)

        e = np.concatenate([x[0:3] - p_ref, x[3:6] - v_ref])
        y_err = e[0:3]

        # stop accumulating once error is too large, or once accumulating
        # further would push the integral term's own contribution past its cap
        if np.linalg.norm(y_err[:2]) < self.e_band:
            xi_trial = self.xi + y_err*dt
            if np.linalg.norm(self.Ki @ xi_trial) <= self.u_i_max:
                self.xi = xi_trial

        u = a_ref - self.K @ e - self.Ki @ self.xi

        return u

    def reset(self):
        self.xi[:] = 0


class OuterLoopPayloadLQR:
    """
    3x10 gain outputs acceleration
    """

    def __init__(self,
                 w_pos_xy=None, w_pos_z=None,
                 tuning_const=None
                 ):
        w_pos_xy = config.LQR_PAYLOAD_W_POS_XY if w_pos_xy is None else w_pos_xy
        w_pos_z = config.LQR_PAYLOAD_W_POS_Z if w_pos_z is None else w_pos_z
        tuning_const = (config.LQR_PAYLOAD_TUNING_CONST if tuning_const is None
                        else tuning_const)
        L = config.TETHER_LEN
        A, B = self._build_system()

        # outputs: payload position (x, y, z) and yaw
        C = np.zeros((3, 10))
        C[0, 0] = 1
        C[0, 6] = L  # payload x = s1 + L*alpha_x

        C[1, 1] = 1
        C[1, 7] = L  # payload y = s2 + L*alpha_y

        C[2, 2] = 1  # payload z = s3

        W = np.diag([w_pos_xy, w_pos_xy, w_pos_z])
        Q = C.T@W@C

        # input cost for the order of [a1, a2, a3]
        R = tuning_const*np.diag([1, 1, 1])
        subK = self._lqr(A, B, Q, R)  # (3, 10)

        self.K = np.zeros((3, 16))
        # states used for designing the LQR controller
        outerLoopStates = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15]
        self.K[:, outerLoopStates] = subK
        self.subK = subK
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q

    def compute_u(self, state_err):
        return -self.K @ state_err

    @staticmethod
    def _build_system():
        """
        linearized 10-state about hover.
        state: [s1 s2 s3  v1 v2 v3  a1 a2  a1d a2d]
        """
        L = config.TETHER_LEN
        g = config.GRAVITY
        a = np.zeros((10, 10))
        b = np.zeros((10, 3))
        # position
        a[0, 3] = 1
        a[1, 4] = 1
        a[2, 5] = 1
        # translational kinematics
        b[3, 0] = 1
        b[4, 1] = 1
        b[5, 2] = 1
        # pendulum kinematics and dynamics
        a[6, 8] = 1
        a[7, 9] = 1
        a[8, 6] = -g/L
        a[9, 7] = -g/L

        b[8, 0] = -1/L
        b[9, 1] = -1/L

        return a, b

    @staticmethod
    def _lqr(A, B, Q, R):
        """Returns LQR gain"""
        P = solve_continuous_are(A, B, Q, R)
        return np.linalg.inv(R) @ B.T @ P


class OuterLoopPayloadLQRWithLag:
    """
    3x16 state gain + 3x3 lag gain, outputs acceleration

    Same design as OuterLoopPayloadLQR with the airframe's acceleration lag in
    the model, so the swing is driven by the acceleration that arrived rather
    than the one that was asked for. The lag state is not measured, it is the
    same first order model run forward on what was already sent.
    """

    def __init__(self,
                 w_pos_xy=None, w_pos_z=None,
                 tuning_const=None, tau=None
                 ):
        w_pos_xy = config.LQR_PAYLOAD_W_POS_XY if w_pos_xy is None else w_pos_xy
        w_pos_z = config.LQR_PAYLOAD_W_POS_Z if w_pos_z is None else w_pos_z
        tuning_const = (config.LQR_PAYLOAD_LAG_TUNING_CONST
                        if tuning_const is None else tuning_const)
        tau = config.ACCEL_LAG_TAU if tau is None else tau
        L = config.TETHER_LEN
        A, B = self._build_system(tau)

        # outputs: payload position (x, y, z)
        C = np.zeros((3, 13))
        C[0, 0] = 1
        C[0, 6] = L  # payload x = s1 + L*alpha_x

        C[1, 1] = 1
        C[1, 7] = L  # payload y = s2 + L*alpha_y

        C[2, 2] = 1  # payload z = s3

        W = np.diag([w_pos_xy, w_pos_xy, w_pos_z])
        Q = C.T@W@C

        # input cost for the order of [a1, a2, a3]
        R = tuning_const*np.diag([1, 1, 1])
        subK = self._lqr(A, B, Q, R)  # (3, 13)

        self.K = np.zeros((3, 16))
        # states used for designing the LQR controller
        outerLoopStates = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15]
        self.K[:, outerLoopStates] = subK[:, :10]
        self.Ka = subK[:, 10:]              # (3, 3) on the accel still arriving
        self.a_hat = np.zeros(3)
        self.subK = subK
        self.A = A
        self.B = B
        self.C = C
        self.R = R
        self.Q = Q
        self.tau = tau

    def compute_u(self, state_err):
        dt = 1/config.CONTROL_FREQUENCY

        u = -self.K @ state_err - self.Ka @ self.a_hat

        # the airframe is still catching up to what was already sent, so run
        # the lag forward on this command for the next tick to answer to
        self.a_hat += (u - self.a_hat)*dt/self.tau

        return u

    def reset(self):
        self.a_hat[:] = 0

    @staticmethod
    def _build_system(tau):
        """
        linearized 13-state about hover, the command reaching the airframe
        through a first order lag.
        state: [s1 s2 s3  v1 v2 v3  a1 a2  a1d a2d  ax ay az]
        """
        L = config.TETHER_LEN
        g = config.GRAVITY
        a = np.zeros((13, 13))
        b = np.zeros((13, 3))
        # position
        a[0, 3] = 1
        a[1, 4] = 1
        a[2, 5] = 1
        # translational kinematics, driven by the acceleration that arrived
        a[3, 10] = 1
        a[4, 11] = 1
        a[5, 12] = 1
        # pendulum kinematics and dynamics
        a[6, 8] = 1
        a[7, 9] = 1
        a[8, 6] = -g/L
        a[9, 7] = -g/L

        a[8, 10] = -1/L
        a[9, 11] = -1/L

        # the airframe chases the command rather than meeting it
        a[10, 10] = -1/tau
        a[11, 11] = -1/tau
        a[12, 12] = -1/tau

        b[10, 0] = 1/tau
        b[11, 1] = 1/tau
        b[12, 2] = 1/tau

        return a, b

    @staticmethod
    def _lqr(A, B, Q, R):
        """Returns LQR gain"""
        P = solve_continuous_are(A, B, Q, R)
        return np.linalg.inv(R) @ B.T @ P


class OuterLoopPayloadLQI:
    """
    3x16 state gain + 3x3 integral gain, outputs acceleration
    """
    OUTER_STATES = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15]

    def __init__(self,
                 w_pos_xy=None, w_pos_z=None,
                 w_int_xy=None, w_int_z=None,
                 tuning_const=None,
                 e_band=None, u_i_max=None):
        w_pos_xy = config.LQI_PAYLOAD_W_POS_XY if w_pos_xy is None else w_pos_xy
        w_pos_z = config.LQI_PAYLOAD_W_POS_Z if w_pos_z is None else w_pos_z
        w_int_xy = config.LQI_PAYLOAD_W_INT_XY if w_int_xy is None else w_int_xy
        w_int_z = config.LQI_PAYLOAD_W_INT_Z if w_int_z is None else w_int_z
        tuning_const = (config.LQI_PAYLOAD_TUNING_CONST if tuning_const is None
                        else tuning_const)
        e_band = config.LQI_PAYLOAD_E_BAND if e_band is None else e_band
        u_i_max = config.LQI_PAYLOAD_U_I_MAX if u_i_max is None else u_i_max
        L = config.TETHER_LEN
        A, B = self._build_system()
        C = np.zeros((3, 10))
        C[0, 0] = 1
        C[0, 6] = L
        C[1, 1] = 1
        C[1, 7] = L
        C[2, 2] = 1

        # augment: xI_dot = C @ x
        Abar = np.zeros((13, 13))
        Abar[:10, :10] = A
        Abar[10:, :10] = C
        Bbar = np.zeros((13, 3))
        Bbar[:10, :] = B

        W = np.diag([w_pos_xy, w_pos_xy, w_pos_z])
        Wi = np.diag([w_int_xy, w_int_xy, w_int_z])
        Q = np.zeros((13, 13))
        Q[:10, :10] = C.T@W@C
        Q[10:, 10:] = Wi
        R = tuning_const*np.diag([1, 1, 1])

        Kbar = self._lqr(Abar, Bbar, Q, R)  # (3, 13)
        self.C = C
        self.K = np.zeros((3, 16))
        self.K[:, self.OUTER_STATES] = Kbar[:, :10]
        self.Ki = Kbar[:, 10:]              # (3, 3)
        self.xi = np.zeros(3)

        # make accessible for analysis
        self.Abar = Abar
        self.Bbar = Bbar
        self.Kbar = Kbar
        self.Q = Q
        self.R = R

        self.e_band = e_band
        self.u_i_max = u_i_max

    def compute_u(self, state_err):
        dt = 1/config.CONTROL_FREQUENCY

        y_err = self.C @ state_err[self.OUTER_STATES]

        # stop accumulating once error is too large, or once accumulating
        # further would push the integral term's own contribution past its cap
        if np.linalg.norm(y_err[:2]) < self.e_band:
            xi_trial = self.xi + y_err*dt
            if np.linalg.norm(self.Ki @ xi_trial) <= self.u_i_max:
                self.xi = xi_trial

        u = -self.K @ state_err - self.Ki @ self.xi
        return u

    def reset(self):
        self.xi[:] = 0

    @staticmethod
    def _build_system():
        """
        linearized 10-state about hover.
        state: [s1 s2 s3  v1 v2 v3  a1 a2  a1d a2d]
        """
        L = config.TETHER_LEN
        g = config.GRAVITY
        a = np.zeros((10, 10))
        b = np.zeros((10, 3))
        # position
        a[0, 3] = 1
        a[1, 4] = 1
        a[2, 5] = 1
        # translational kinematics
        b[3, 0] = 1
        b[4, 1] = 1
        b[5, 2] = 1
        # pendulum kinematics and dynamics
        a[6, 8] = 1
        a[7, 9] = 1
        a[8, 6] = -g/L
        a[9, 7] = -g/L

        b[8, 0] = -1/L
        b[9, 1] = -1/L

        return a, b

    @staticmethod
    def _lqr(A, B, Q, R):
        """Returns LQR gain"""
        P = solve_continuous_are(A, B, Q, R)
        return np.linalg.inv(R) @ B.T @ P


class OuterLoopPayloadLQIWithLag:
    """
    3x16 state gain + 3x3 integral gain + 3x3 lag gain, outputs acceleration
    """
    OUTER_STATES = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15]

    def __init__(self,
                 w_pos_xy=None, w_pos_z=None,
                 w_int_xy=None, w_int_z=None,
                 tuning_const=None,
                 e_band=None, u_i_max=None, tau=None):
        w_pos_xy = config.LQI_PAYLOAD_W_POS_XY if w_pos_xy is None else w_pos_xy
        w_pos_z = config.LQI_PAYLOAD_W_POS_Z if w_pos_z is None else w_pos_z
        w_int_xy = config.LQI_PAYLOAD_W_INT_XY if w_int_xy is None else w_int_xy
        w_int_z = config.LQI_PAYLOAD_W_INT_Z if w_int_z is None else w_int_z
        tuning_const = (config.LQI_PAYLOAD_LAG_TUNING_CONST
                        if tuning_const is None else tuning_const)
        e_band = config.LQI_PAYLOAD_E_BAND if e_band is None else e_band
        u_i_max = config.LQI_PAYLOAD_U_I_MAX if u_i_max is None else u_i_max
        tau = config.ACCEL_LAG_TAU if tau is None else tau
        L = config.TETHER_LEN
        A, B = self._build_system(tau)

        # payload position, off the states the drone and swing share
        C = np.zeros((3, 13))
        C[0, 0] = 1
        C[0, 6] = L
        C[1, 1] = 1
        C[1, 7] = L
        C[2, 2] = 1

        # augment: xI_dot = C @ x
        Abar = np.zeros((16, 16))
        Abar[:13, :13] = A
        Abar[13:, :13] = C
        Bbar = np.zeros((16, 3))
        Bbar[:13, :] = B

        W = np.diag([w_pos_xy, w_pos_xy, w_pos_z])
        Wi = np.diag([w_int_xy, w_int_xy, w_int_z])
        Q = np.zeros((16, 16))
        Q[:13, :13] = C.T@W@C
        Q[13:, 13:] = Wi
        R = tuning_const*np.diag([1, 1, 1])

        Kbar = self._lqr(Abar, Bbar, Q, R)  # (3, 16)
        self.C = C[:, :10]                  # the lag states carry no output
        self.K = np.zeros((3, 16))
        self.K[:, self.OUTER_STATES] = Kbar[:, :10]
        # (3, 3) on the accel still arriving
        self.Ka = Kbar[:, 10:13]
        self.Ki = Kbar[:, 13:]              # (3, 3)
        self.xi = np.zeros(3)
        self.a_hat = np.zeros(3)

        # make accessible for analysis
        self.Abar = Abar
        self.Bbar = Bbar
        self.Kbar = Kbar
        self.Q = Q
        self.R = R

        self.e_band = e_band
        self.u_i_max = u_i_max
        self.tau = tau

    def compute_u(self, state_err):
        dt = 1/config.CONTROL_FREQUENCY

        y_err = self.C @ state_err[self.OUTER_STATES]

        # stop accumulating once error is too large, or once accumulating
        # further would push the integral term's own contribution past its cap
        if np.linalg.norm(y_err[:2]) < self.e_band:
            xi_trial = self.xi + y_err*dt
            if np.linalg.norm(self.Ki @ xi_trial) <= self.u_i_max:
                self.xi = xi_trial

        u = -self.K @ state_err - self.Ki @ self.xi - self.Ka @ self.a_hat

        # the airframe is still catching up to what was already sent, so run
        # the lag forward on this command for the next tick to answer to
        self.a_hat += (u - self.a_hat)*dt/self.tau

        return u

    def reset(self):
        self.xi[:] = 0
        self.a_hat[:] = 0

    @staticmethod
    def _build_system(tau):
        """
        linearized 13-state about hover, the command reaching the airframe
        through a first order lag.
        state: [s1 s2 s3  v1 v2 v3  a1 a2  a1d a2d  ax ay az]
        """
        L = config.TETHER_LEN
        g = config.GRAVITY
        a = np.zeros((13, 13))
        b = np.zeros((13, 3))
        # position
        a[0, 3] = 1
        a[1, 4] = 1
        a[2, 5] = 1
        # translational kinematics, driven by the acceleration that arrived
        a[3, 10] = 1
        a[4, 11] = 1
        a[5, 12] = 1
        # pendulum kinematics and dynamics
        a[6, 8] = 1
        a[7, 9] = 1
        a[8, 6] = -g/L
        a[9, 7] = -g/L

        a[8, 10] = -1/L
        a[9, 11] = -1/L

        # the airframe chases the command rather than meeting it
        a[10, 10] = -1/tau
        a[11, 11] = -1/tau
        a[12, 12] = -1/tau

        b[10, 0] = 1/tau
        b[11, 1] = 1/tau
        b[12, 2] = 1/tau

        return a, b

    @staticmethod
    def _lqr(A, B, Q, R):
        """Returns LQR gain"""
        P = solve_continuous_are(A, B, Q, R)
        return np.linalg.inv(R) @ B.T @ P


class PositionController:
    """
    Alternative controller: returns acceleration from a spring-mass damper system
    """

    def __init__(self, wn_xy=0.4, wn_z=1.2, zeta=1):
        self.wn = np.array([wn_xy, wn_xy, wn_z])
        self.zeta = zeta

    def compute_u(self, x, p_ref, v_ref, a_ref=None):
        p_s = np.asarray(p_ref)
        v_s = np.asarray(v_ref)

        p_D = x[0:3]
        v_D = x[3:6]

        if a_ref is None:
            a_ref = np.zeros(3)

        u = a_ref + 2*self.zeta*self.wn * \
            (v_s - v_D) + self.wn**2*(p_s - p_D)

        return u


class ArduPilotFlightController:
    """
    "ME236" classic cascaded control system for the inner-loop

    accepts an acceleration and yaw setpoint valid for SET_POSITION_TARGET_LOCAL_NED
    """

    def __init__(self,
                 tau_phi=0.3,
                 tau_theta=0.3,
                 tau_psi=0.5,
                 tau_p=0.05,
                 tau_q=0.05,
                 tau_r=0.08):

        self.tau_theta = tau_theta
        self.tau_phi = tau_phi
        self.tau_psi = tau_psi
        self.tau_p = tau_p
        self.tau_q = tau_q
        self.tau_r = tau_r

    def compute_u(self, x, a_des, yaw_s):
        """
        Computes [C_Sigma, n1, n2, n3] from the acceleration and yaw setpoint
        """
        g = config.GRAVITY
        m = config.MASS_TOTAL
        Jx = config.J[0, 0]
        Jz = config.J[2, 2]

        phi = x[6]
        theta = x[7]
        psi = x[8]
        p = x[9]
        q = x[10]
        r = x[11]

        a1, a2, a3 = a_des

        # thrust calculation
        C_Sigma = m*np.sqrt(a1**2 + a2**2 + (a3 + g)**2)

        # acceleration transform
        theta_s = a1*m/C_Sigma
        phi_s = -a2*m/C_Sigma

        # attitude control
        p_s = (1/self.tau_phi)*(phi_s - phi)
        q_s = (1/self.tau_theta)*(theta_s - theta)
        r_s = (1/self.tau_psi)*(yaw_s - psi)

        # angular velocity control
        n1 = (Jx/self.tau_p)*(p_s - p) + (Jx - Jz)*q*r
        n2 = (Jx/self.tau_q)*(q_s - q) + (Jz - Jx)*p*r
        n3 = (Jz/self.tau_r)*(r_s - r)

        return [C_Sigma, n1, n2, n3]
