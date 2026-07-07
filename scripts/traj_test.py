import sim.drone_test_only_mission_manager as mission
import sim.SITL_dynamics as dynamics

import comms.common as comms
import comms.control as control
import time

from pymavlink import mavutil


def check_rates(m, seconds=3):
    from collections import Counter
    counts = Counter()
    t_end = time.time() + seconds
    while time.time() < t_end:
        msg = m.recv_match(blocking=True, timeout=0.5)
        if msg:
            counts[msg.get_type()] += 1
    for name, n in sorted(counts.items()):
        print(f"{name:28s} {n / seconds:6.1f} Hz")


def set_guid_options(m, bitmask=48):
    """
    set GUID_OPTIONS to the specified bitmask
    """
    m.mav.param_set_send(m.target_system,
                         m.target_component,
                         b"GUID_OPTIONS",
                         48,
                         mavutil.mavlink.MAV_PARAM_TYPE_INT32)


if __name__ == '__main__':
    # connection = "/dev/ttyACM0"
    # baud = 115200
    connection = "udp:127.0.0.1:14550"
    takeoff_altitude = 10
    control_freq = 50

    # add baud here if connected to real drone
    m = comms.connect(connection)

    comms.wait_until_armable(m)
    # comms.set_mode(m, "LOITER")
    set_guid_options(m, 48)
    comms.set_mode(m, "GUIDED")
    comms.arm(m)
    comms.takeoff(m, takeoff_altitude)

    control.request_fast_state(m, hz=control_freq)

    # debugging
    time.sleep(3)
    check_rates(m)

    # straight line
    controller = dynamics.OuterLoopLQR()
    speed = 0.5

    # 20 m test
    ref = mission.ReferenceTrajectory([0, 0, 10], [20, 0, 10], speed=speed)

    print("running custom controller...")
    control.fly_trajectory(
        m, ref, controller, duration=ref.total_time_to_wp, dt=1/control_freq, reassert=False)

    # debugging
    check_rates(m)

    comms.land(m)
