import csv
import os
import time
from datetime import datetime
import numpy as np


class FlightLogger:
    def __init__(self, data_dir="data"):
        os.makedirs(data_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(data_dir, f"flight_{stamp}.csv")
        self.f = open(self.path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow([
            "wall_time", "t", "loop_dt", "mode", "armed",
            "px_ref", "py_ref", "pz_ref", "vx_ref", "vy_ref", "vz_ref",
            "px", "py", "pz", "vx", "vy", "vz",
            "roll", "pitch", "yaw",
            "ex", "ey", "ez", "pos_err_norm",
            "ux", "uy", "uz", "u_norm",
            "batt_v",
        ])  # all positions ENU, meters frame stated here on purpose
        self._last = time.time()
        self._n = 0

    def log(self, t, mode, armed, p_ref, v_ref, x, u, rpy, batt_v):
        now = time.time()
        dt, self._last = now - self._last, now
        p, v = x[0:3], x[3:6]
        e = p - p_ref
        self.w.writerow([
            f"{now:.4f}", f"{t:.4f}", f"{dt:.4f}", mode, int(armed),
            *p_ref, *v_ref, *p, *v, *rpy,
            *e, np.linalg.norm(e), *u, np.linalg.norm(u), batt_v,
        ])
        self.f.flush()  # survive a crash mid-flight
        self._n += 1
        if self._n % 250 == 0:  # force to disk every 10s at 25Hz
            os.fsync(self.f.fileno())

    def close(self):
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()
