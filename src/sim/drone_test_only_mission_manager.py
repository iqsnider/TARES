import numpy as np

EARTH_R = 6378137

def geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0):
    """
    Convers lat lon to enu
    """
    dlat = np.radians(lat - lat0)
    dlon = np.radians(lon - lon0)
    east = dlon*EARTH_R*np.cos(np.radians(lat0))
    north = dlat*EARTH_R
    up = alt - alt0
    
    return np.array([east, north, up])

def read_wpl_waypoints(path):
    """
    Parse a mission planner wp file
    """
    pts = []
    with open(path) as f:
        lines = f.read().splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        c = line.split("\t")
        pts.append((int(c[3]), float(c[8]), float(c[9]), float(c[10])))

    return pts

class ReferenceTrajectory:
    """
    Pick a t, get a (p_ref, v_ref)
    """

    def __init__(self, p_start, p_end, speed):

        # start position of ref point
        self.p0 = np.asarray(p_start, dtype=float)
        p1 = np.asarray(p_end, dtype=float)
        delta = p1 - self.p0
        dist = np.linalg.norm(delta)
        self.total_time_to_wp = dist / speed  # time to reach B
        self.v_const = delta / self.total_time_to_wp  # constant velocity vector

    def __call__(self, t):
        """
        Return (p_ref, v_ref) at specified time.
        """
        if t < self.total_time_to_wp:
            p_ref = self.p0 + self.v_const*t
            v_ref = self.v_const.copy()
            return p_ref, v_ref

        p_ref = self.p0 + self.v_const*self.total_time_to_wp
        v_ref = np.zeros(3)
        return p_ref, v_ref


class MultiPointReferenceTrajectory:
    """
    Chains together multiple straight line reference trajectories from a collection of points
    """
    def __init__(self, points, speed):
        self.segments = [ReferenceTrajectory(points[i], points[i+1], speed) for i in range(len(points)-1)]
        self.total_time = sum(seg.total_time_to_wp for seg in self.segments)

    def __call__(self, t):
        """
        Return (p_ref, v_ref) at a specified time.
        """
        for seg in self.segments:
            if t <= seg.total_time_to_wp:
                return seg(t)
            t -= self.total_time_to_wp
        last = self.segments[-1]
        return last.p1.copy(), np.zeros(3)


def trajectory_from_waypoint_file(path, speed, origin):
    """generate the trajectory chain from a given waypoint planner file"""

    wps = read_qgc_waypoints(path)
    lat0, lon0, alt0 = (origin if origin is not None else (wps[0][1], wps[0][2], wps[0][3]))
    nav = [(lat, lon, alt) for (cmd, lat, lon, alt) in wps if cmd == 16]

    pts = [geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0) for (lat, lon, alt) in nav]

    return MultiPointReferenceTrajectory(pts, speed)


if __name__ == "__main__":
    trajectory = ReferenceTrajectory([0,0,10], [100,0,10], speed=1)
    for t in np.linspace(0, trajectory.total_time_to_wp,9):
        p, v = trajectory(t)
        print(f"t={t:5.1f}, pos={p[0]:6.1f}, vel={v[0]:5.2f}")
