from pymavlink import mavutil
import time
import os
import csv
from datetime import datetime
import numpy as np
import comms.common as comms
import comms.control as control

M = mavutil.mavlink

if __name__ == '__main__':
    # connection = "/dev/ttyACM0"
    # baud = 115200
    connection = "udp:127.0.0.1:14550"
    takeoff_altitude = 10

    m = comms.connect(connection)
    comms.wait_until_armable(m)
    comms.set_mode(m, "GUIDED")
    comms.arm(m)
    comms.takeoff(m, takeoff_altitude)
    control.request_fast_state(m, hz=25)

    # --- set up the smoke-test log ---
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", f"smoke_{datetime.now():%Y%m%d_%H%M%S}.csv")
    f = open(path, "w", newline="")
    w = csv.writer(f)
    w.writerow(["t", "cmd_az_ned", "alt_agl", "vz_enu", "az_meas_enu"])

    print("1 m/s^2 up for 3 s")
    cmd = np.array([0, 0, -1.0])           # NED, -Z = up
    t0 = time.time()
    x = None
    try:
        while (t := time.time() - t0) < 3:
            control.send_accel(m, cmd)
            x = control.get_state_enu(m, prev=x)   # [p_enu(3), v_enu(3)]
            if x is not None:
                alt = x[2]                          # ENU up = AGL-ish
                vz = x[5]                           # ENU vertical velocity
                # measured accel: pull from IMU if you want it, else leave blank
                w.writerow([f"{t:.3f}", cmd[2], f"{alt:.3f}", f"{vz:.3f}", ""])
                f.flush()
            time.sleep(0.04)
    finally:
        f.close()
        print(f"log -> {path}")

    comms.land(m)
