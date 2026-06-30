from pymavlink import mavutil
import time
import numpy as np

import sim.config as config
import sim.SITL_dynamics as dynamics

import comms.common as comms
import comms.control as control


if __name__ == '__main__':
    # connection = "/dev/ttyACM0"
    # baud = 115200
    connection = "udp:127.0.0.1:14550"
    takeoff_altitude = 10

    # add baud here if connected to actual drone
    m = comms.connect(connection)

    comms.wait_until_armable(m)
    comms.set_mode(m, "GUIDED")
    comms.arm(m)
    comms.takeoff(m, takeoff_altitude)

    control.request_fast_state(m, hz=25)

    print("1 m/s^2 up for 3 s")
    t0 = time.time()
    while time.time() - t0 < 3:
        control.send_accel(m, np.array([0, 0, -1]))  # NED -Z = up
        time.sleep(0.04)

    comms.land(m)
