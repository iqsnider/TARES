import numpy as np

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
