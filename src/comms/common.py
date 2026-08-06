from pymavlink import mavutil
from collections import Counter
import time




def connect(connection, baud=None):
    print(f"connecting to {connection}")
    # setup listener on the specified port
    if baud is not None:
        the_connection = mavutil.mavlink_connection(connection, baud)
    else:
        the_connection = mavutil.mavlink_connection(connection)

    the_connection.wait_heartbeat()

    print(f"connected to {connection}")

    return the_connection


def wait_until_armable(m, timeout=60):
    """
    Waits for GPS and EKF to stabilize
    """
    t0 = time.time()
    print("waiting until armable")
    while time.time() - t0 < timeout:
        msg = m.recv_match(
            type="GPS_RAW_INT", blocking=True, timeout=2)

        if msg and msg.fix_type >= 3:
            print("armable")
            return
    raise TimeoutError("EKF/GPS never stabilized")


def set_mode(m, mode_name, timeout=5):
    """
    Set the copter mode

    Gives up rather than waiting forever. This gets called from cleanup paths
    with the vehicle still in the air, where a command the autopilot never
    acts on would otherwise wedge the script mid-flight and take the flight
    data down with it.
    """
    # the heartbeat tells us the vehicle type which ensures that an unknown mode does not get enabled
    if m.wait_heartbeat(timeout=timeout) is None:
        raise TimeoutError(f"no heartbeat within {timeout}s, cannot set {mode_name}")

    mode_id = m.mode_mapping()[mode_name]
    m.mav.set_mode_send(m.target_system,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                        mode_id)

    deadline = time.time() + timeout
    while time.time() < deadline:
        heartbeat = m.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
        if heartbeat and heartbeat.custom_mode == mode_id:
            print(f"mode = {mode_name}")
            return
    raise TimeoutError(f"mode did not change to {mode_name} within {timeout}s")


def arm(m):
    """
    attempts to arm ardupilot
    """
    for attempt in range(20):
        m.mav.command_long_send(m.target_system,
                                m.target_component,
                                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                                0, 1, 0, 0, 0, 0, 0, 0)
        ack = m.recv_match(
            type="COMMAND_ACK", blocking=True, timeout=3)

        if ack and ack.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
            if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                m.motors_armed_wait()
                print("armed")
                return

        print(f"arm rejected (result={ack.result}); retrying...")
        time.sleep(2)

    raise RuntimeError("Could not arm")


def takeoff(m, altitude):
    """
    commands vehicle takeoff to a specified altitutde
    """
    m.mav.command_long_send(m.target_system,
                            m.target_component,
                            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                            0, 0, 0, 0, 0, 0, 0,
                            altitude)
    print(f"taking off to {altitude}m")
    while True:
        msg = m.recv_match(type="GLOBAL_POSITION_INT", blocking=True)
        rel = msg.relative_alt/1000  # m
        if rel >= altitude*0.95:
            print(f"reached hover altitude {rel:.2f}m")
            return


def hover(m, seconds):
    """
    Commands ardupilot to hover for a specified amount of time
    """
    t0 = time.time()
    while time.time() - t0 < seconds:
        msg = m.recv_match(
            type="GLOBAL_POSITION_INT", blocking=True, timeout=1)

        if msg:
            print(f"{msg.relative_alt/1000:5.2f}")

    print("done hovering")


def land(m):
    """
    important
    """
    set_mode(m, "LAND")
    m.motors_disarmed_wait()
    print("landed and disarmed")

def check_rates(m, seconds=3):
    """
    Checks the data stream rate to make sure it's not being overwritten by a proxy on the same channel
    """
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
                         bitmask,
                         mavutil.mavlink.MAV_PARAM_TYPE_INT32)
