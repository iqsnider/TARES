import numpy as np

import Prm.config as config

# EARTH_R = 6378137 # [m]
#
#
# def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
#     """
#     TODO: work in progress
#     Convers lat lon to enu
#     """
#     dlat = np.radians(lat - lat0)
#     dlon = np.radians(lon - lon0)
#     east = dlon*EARTH_R*np.cos(np.radians(lat0))
#     north = dlat*EARTH_R
#     up = alt - alt0
#
#     return np.array([east, north, up])
#
#
# def read_wpl_waypoints(path):
#     """
#     TODO: work in progress
#     Parse a mission planner wp file
#     """
#     pts = []
#     with open(path) as f:
#         lines = f.read().splitlines()
#     for line in lines[1:]:
#         if not line.strip():
#             continue
#         c = line.split("\t")
#         pts.append((int(c[3]), float(c[8]), float(c[9]), float(c[10])))
#
#     return pts


class RampedTrajectory:
    """
    ENU. Same interface as ReferenceTrajectory, but the speed ramps in and out
    at a bounded acceleration instead of stepping, the way the stick reference
    slews to the pilot's command.

    Trapezoidal along the straight line from p_start to p_end: accelerate at
    accel up to speed, cruise, then decelerate onto the endpoint at rest. A leg
    too short to reach speed gets a triangular profile instead.
    """

    def __init__(self, p_start, p_end, speed, startPointHoverTime=1, endPointHoverTime=1, accel=None):
        self.p0 = np.asarray(p_start, dtype=float)
        self.p1 = np.asarray(p_end, dtype=float)
        delta = self.p1 - self.p0
        self.dist = np.linalg.norm(delta)
        self.u = delta / self.dist
        self.accel = config.MAX_STICK_ACCELERATION if accel is None else accel

        # a leg shorter than both ramps never reaches speed, so it tops out
        # where the two ramps meet
        ramp_dist = speed**2 / (2*self.accel)
        if 2*ramp_dist <= self.dist:
            self.v_peak = speed
        else:
            self.v_peak = np.sqrt(self.accel*self.dist)
            ramp_dist = self.v_peak**2 / (2*self.accel)

        self.ramp_dist = ramp_dist
        self.t_ramp = self.v_peak / self.accel
        self.t_cruise = (self.dist - 2*ramp_dist) / self.v_peak
        self.total_time_to_wp = 2*self.t_ramp + self.t_cruise

        self.startPointHoverTime = startPointHoverTime
        self.endPointHoverTime = endPointHoverTime

        # phase boundaries
        self.t_move_start = startPointHoverTime
        self.t_move_end = startPointHoverTime + self.total_time_to_wp

        # total time parameter
        self.duration = startPointHoverTime + self.total_time_to_wp + endPointHoverTime

    def __call__(self, t):
        """
        Return (p_ref, v_ref) at specified time.
        """
        # hover at start
        if t < self.t_move_start:
            return self.p0.copy(), np.zeros(3)

        # hover at end
        if t >= self.t_move_end:
            return self.p1.copy(), np.zeros(3)

        tau = t - self.t_move_start

        # ramping up to speed
        if tau < self.t_ramp:
            s = 0.5*self.accel*tau**2
            sdot = self.accel*tau

        # holding speed
        elif tau < self.t_ramp + self.t_cruise:
            s = self.ramp_dist + self.v_peak*(tau - self.t_ramp)
            sdot = self.v_peak

        # ramping down onto the endpoint, written from the far end so it
        # arrives at the waypoint exactly at rest
        else:
            t_left = self.total_time_to_wp - tau
            s = self.dist - 0.5*self.accel*t_left**2
            sdot = self.accel*t_left

        p_ref = self.p0 + s*self.u
        v_ref = sdot*self.u

        return p_ref, v_ref


class ReferenceTrajectory:
    """
    ENU. A bare bones reference trajectory. This class should be inherited/passed into a wrapper class to ensure vehicle safety.
    Pick a t, get a (p_ref, v_ref)
    """

    def __init__(self, p_start, p_end, speed, startPointHoverTime=1, endPointHoverTime=1):
        self.p0 = np.asarray(p_start, dtype=float)
        self.p1 = np.asarray(p_end, dtype=float)
        delta = self.p1 - self.p0
        dist = np.linalg.norm(delta)
        self.total_time_to_wp = dist / speed
        self.v_const = delta / self.total_time_to_wp
        self.startPointHoverTime = startPointHoverTime
        self.endPointHoverTime = endPointHoverTime
        # phase boundaries
        self.t_move_start = startPointHoverTime
        self.t_move_end = startPointHoverTime + self.total_time_to_wp

        # total time parameter
        self.duration = startPointHoverTime + self.total_time_to_wp + endPointHoverTime

    def __call__(self, t):
        """
        Return (p_ref, v_ref) at specified time.
        """
        # hover at start
        if t < self.t_move_start:
            return self.p0.copy(), np.zeros(3)

        # moving at constant velocity
        elif t < self.t_move_end:
            tau = t - self.t_move_start
            p_ref = self.p0 + self.v_const * tau
            return p_ref, self.v_const.copy()

        # hover at end
        else:
            return self.p1.copy(), np.zeros(3)
