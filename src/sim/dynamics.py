import math
import numpy as np
from scipy.linalg import solve_continuous_are
import sim.config as config

HOVER_THRUST = config.MASS_TOTAL*config.GRAVITY


def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


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
                 q_pos_xy=1, q_pos_z=1,
                 q_vel_xy=1, q_vel_z=1,
                 r_acc=1):
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



class OuterLoopPayloadLQR:
    """
    4x10 gain outputs acceleration
    """

    def __init__(self,
                 w_pos_xy=(1/1)**2,
                 w_pos_z=(1/1)**2,
                 tuning_const=1/1**2
                 ):
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
        g = GRAVITY
        m = MASS_TOTAL
        Jx = J[0, 0]
        Jz = J[2, 2]

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
