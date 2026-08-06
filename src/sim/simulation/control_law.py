import numpy as np

import Prm.config as config
import sim.dynamics as dynamics
from sim.dynamics import tether_equilibrium_state

# architecture used when none is requested
DEFAULT_ARCHITECTURE = 'payload_lqr'

# yaw setpoint [rad]
YAW_REF = 0


def _drone_reference(ref, t):
    """Drone position/velocity setpoint that hangs the payload on the ref."""
    p_pl, v_pl = ref(t)
    return p_pl + np.array([0, 0, config.TETHER_LEN]), v_pl


class PayloadLQR:
    """
    Outer loop: 10-state payload LQR (swing aware) -> acceleration command
    Inner loop: ArduPilot-style cascaded attitude/rate controller
    """

    def __init__(self, w_pos_xy=(1/1)**2, w_pos_z=(1/0.1)**2, tuning_const=1):
        self.outer = dynamics.OuterLoopPayloadLQR(w_pos_xy=w_pos_xy,
                                         w_pos_z=w_pos_z,
                                         tuning_const=tuning_const)
        self.inner = dynamics.ArduPilotFlightController()

    def __call__(self, x, t, ref):
        p_pl_ref, v_pl_ref = ref(t)
        x_star = tether_equilibrium_state(p_pl_ref, v_pl_ref,
                                          config.TETHER_LEN)
        e = x - x_star
        e[6:9] = dynamics.wrap_angle(e[6:9])

        a_des = self.outer.compute_u(e)
        u = self.inner.compute_u(x, a_des, yaw_s=YAW_REF)
        return np.asarray(u, dtype=float), e


class DroneLQR:
    """
    Outer loop: 6-state double-integrator LQR on the drone only (swing blind)
    Inner loop: ArduPilot-style cascaded attitude/rate controller
    """

    def __init__(self, q_pos_xy=1.0, q_pos_z=1.0,
                 q_vel_xy=1.0, q_vel_z=1.0, r_acc=1.0):
        self.outer = dynamics.OuterLoopLQR(q_pos_xy=q_pos_xy, q_pos_z=q_pos_z,
                                  q_vel_xy=q_vel_xy, q_vel_z=q_vel_z,
                                  r_acc=r_acc)
        self.inner = dynamics.ArduPilotFlightController()

    def __call__(self, x, t, ref):
        p_ref, v_ref = _drone_reference(ref, t)
        x_star = tether_equilibrium_state(*ref(t), config.TETHER_LEN)
        e = x - x_star
        e[6:9] = dynamics.wrap_angle(e[6:9])

        a_des = self.outer.compute_u(x, p_ref, v_ref)
        u = self.inner.compute_u(x, a_des, yaw_s=YAW_REF)
        return np.asarray(u, dtype=float), e


class DronePD:
    """
    Outer loop: spring-mass-damper position controller on the drone
    Inner loop: ArduPilot simulated cascaded attitude/rate controller
    """

    def __init__(self, wn_xy=0.4, wn_z=1.2, zeta=1.0):
        self.outer = dynamics.PositionController(wn_xy=wn_xy, wn_z=wn_z, zeta=zeta)
        self.inner = dynamics.ArduPilotFlightController()

    def __call__(self, x, t, ref):
        p_ref, v_ref = _drone_reference(ref, t)
        x_star = tether_equilibrium_state(*ref(t), config.TETHER_LEN)
        e = x - x_star
        e[6:9] = dynamics.wrap_angle(e[6:9])

        a_des = self.outer.compute_u(x, p_ref, v_ref)
        u = self.inner.compute_u(x, a_des, yaw_s=YAW_REF)
        return np.asarray(u, dtype=float), e


ARCHITECTURES = {'payload_lqr': PayloadLQR,
                 'drone_lqr': DroneLQR,
                 'drone_pd': DronePD}


def build(name=None, **kwargs):
    name = name or DEFAULT_ARCHITECTURE
    return ARCHITECTURES[name](**kwargs)
