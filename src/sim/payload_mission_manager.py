import numpy as np
from sim.config import (GRID_X_START, GRID_Y_START,
                        GRID_X_END, GRID_ALTITUDE,
                        CRUISE_SPEED)


class ReferenceTrajectory:
    """
    Pick a t, get a (p_ref, v_ref)
    """

    def __init__(self, p_start=None, p_end=None, speed=CRUISE_SPEED):
        if p_start is None:
            p_start = [GRID_X_START, GRID_Y_START, GRID_ALTITUDE]
        if p_end is None:
            p_end = [GRID_X_END, GRID_Y_START, GRID_ALTITUDE]

        # start position of ref point
        self.p0 = np.asarray(p_start, dtype=float)
        p1 = np.asarray(p_end, dtype=float)
        delta = p1 - self.p0
        dist = np.linalg.norm(delta)
        self.total_time_to_wp = dist / speed  # time to reach B
        self.v_const = delta / self.total_time_to_wp  # constant velocity vector

    def __call__(self, t):
        """
        Return (p_PL_ref, v_PL_ref) at specified time.
        """
        if t < self.total_time_to_wp:
            p_PL_ref = self.p0 + self.v_const*t
            v_PL_ref = self.v_const.copy()
            return p_PL_ref, v_PL_ref

        p_PL_ref = self.p0 + self.v_const*self.total_time_to_wp
        v_PL_ref = np.zeros(3)
        return p_PL_ref, v_PL_ref


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
