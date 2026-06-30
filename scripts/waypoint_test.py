import sim.config as config
import sim.drone_test_only_mission_manager as mission
import sim.SITL_dynamics as dynamics

import comms.common as comms
import comms.control as control

import os
import numpy as np


if __name__ == '__main__':
    # connection = "/dev/ttyACM0"
    # baud = 115200
    connection = "udp:127.0.0.1:14550"
    takeoff_altitude = 10

    # add baud here if connected to real drone
    m = comms.connect(connection)

    # validate path on the ground = safety
    path = os.path.expanduser(
        "~/TARES_SITL/waypoints/simtest.waypoints")
    ref, total_time = mission.trajectory_from_waypoint_file(path, speed=0.5)

    # dry-run check before committing to flight
    p0, _ = ref(0)
    print(f"trajectory start (ENU): {p0},  total_time: {total_time:.1f}s")
    for tt in np.linspace(0, total_time, 6):
        p, v = ref(tt)
        print(f"  t={tt:6.1f}  p={p}  |v|={np.linalg.norm(v):.2f}")

    comms.wait_until_armable(m)
    comms.set_mode(m, "GUIDED")
    comms.arm(m)
    comms.takeoff(m, takeoff_altitude)

    control_freq = 50
    control.request_fast_state(m, hz=control_freq)

    # straight line
    controller = dynamics.PositionController()
    speed = 0.5
    control.fly_trajectory(
        m, ref, controller, duration=total_time, dt=1/control_freq)

    comms.land(m)
