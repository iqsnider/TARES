import math
import numpy as np
from scipy.linalg import solve_continuous_are
from sim.config import (MASS_TOTAL, MASS_DRONE, GRAVITY, J, D, K_M)

HOVER_THRUST = MASS_TOTAL*GRAVITY


def wrap_angle(a):
    return (a + np.pi) % (2*np.pi) - np.pi


# def _build_system():
#     """
#     linearized 10-state about hover.
#     state: [s1 s2 s3  v1 v2 v3  a1 a2  a1d a2d]
#     """
#     L = TETHER_LEN
#     g = GRAVITY
#     a = np.zeros((10, 10))
#     b = np.zeros((10, 3))
#     # position
#     a[0, 3] = 1
#     a[1, 4] = 1
#     a[2, 5] = 1
#     # translational kinematics
#     b[3, 0] = 1
#     b[4, 1] = 1
#     b[5, 2] = 1
#     # pendulum kinematics and dynamics
#     a[6, 8] = 1
#     a[7, 9] = 1
#     a[8, 6] = -g/L
#     a[9, 7] = -g/L
#
#     b[8, 0] = -1/L
#     b[9, 1] = -1/L
#
#     return a, b
#
#
# def _lqr(A, B, Q, R):
#     """Returns LQR gain"""
#     P = solve_continuous_are(A, B, Q, R)
#     return np.linalg.inv(R) @ B.T @ P
#
#
# class OuterLoopPayloadLQR:
#     """
#     4x10 gain outputs acceleration
#     """
#
#     def __init__(self,
#                  w_pos_xy=(1/1)**2,
#                  w_pos_z=(1/1)**2,
#                  tuning_const=1/1**2,
#                  moment_arm=D,
#                  thrust_to_torque=K_M):
#         L = TETHER_LEN
#         A, B = _build_system()
#
#         # outputs: payload position (x, y, z) and yaw
#         C = np.zeros((3, 10))
#         C[0, 0] = 1
#         C[0, 6] = L  # payload x = s1 + L*alpha_x
#
#         C[1, 1] = 1
#         C[1, 7] = L  # payload y = s2 + L*alpha_y
#
#         C[2, 2] = 1  # payload z = s3
#
#         W = np.diag([w_pos_xy, w_pos_xy, w_pos_z])
#         Q = C.T@W@C
#
#         # input cost for the order of [a1, a2, a3]
#         R = tuning_const*np.diag([1, 1, 1])
#         subK = _lqr(A, B, Q, R)  # (3, 10)
#
#         self.K = np.zeros((3, 16))
#         # states used for designing the LQR controller
#         outerLoopStates = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15]
#         self.K[:, outerLoopStates] = subK
#
#     def compute_u(self, state_err):
#         return -self.K @ state_err


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
