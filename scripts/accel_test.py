from pymavlink import mavutil
import time
import numpy as np

import sim.config as config
import sim.drone_test_only_mission_manager as mission
import sim.SITL_dynamics as dynamics

M = mavutil.mavlink

ACCEL_ONLY = (M.POSITION_TARGET_TYPEMASK_X_IGNORE
            | M.POSITION_TARGET_TYPEMASK_Y_IGNORE
            | M.POSITION_TARGET_TYPEMASK_Z_IGNORE
            | M.POSITION_TARGET_TYPEMASK_VX_IGNORE
            | M.POSITION_TARGET_TYPEMASK_VY_IGNORE
            | M.POSITION_TARGET_TYPEMASK_VZ_IGNORE
            | M.POSITION_TARGET_TYPEMASK_YAW_IGNORE
            | M.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)

def connect(connection):
    # setup listener on the specified port
    the_connection = mavutil.mavlink_connection(connection)
    the_connection.wait_heartbeat()

    # stream telemetry at 4Hz
    the_connection.mav.request_data_stream_send(the_connection.target_system,
                                                the_connection.target_component,
                                                mavutil.mavlink.MAV_DATA_STREAM_ALL, 4, 1)

    return the_connection


def wait_until_armable(m, timeout=30):
    """Waits for GPS and EKF to stabilize"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        msg = m.recv_match(
            type="GPS_RAW_INT", blocking=True, timeout=2)

        if msg and msg.fix_type >= 3:
            return
    raise TimeoutError("EKF/GPS never stabilized")


def set_mode(m, mode_name):
    mode_id = m.mode_mapping()[mode_name]
    m.mav.set_mode_send(m.target_system,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                        mode_id)

    while True:
        heartbeat = m.recv_match(type="HEARTBEAT", blocking=True)
        if heartbeat.custom_mode == mode_id:
            print(f"mode = {mode_name}")
            return


def arm(m):
    """attempts to arm ardupilot"""

    for attempt in range(10):
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
    """commands vehicle takeoff to a specified altitutde"""
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
    """commands ardupilot to hover for a specified amount of time"""
    t0 = time.time()
    while time.time() - t0 < seconds:
        msg = m.recv_match(
            type="GLOBAL_POSITION_INT", blocking=True, timeout=1)

        if msg:
            print(f"{msg.relative_alt/1000:5.2f}")

    print("done hovering")


def land(m):
    """important"""
    set_mode(m, "LAND")
    m.motors_disarmed_wait()
    print("landed and disarmed")

#######################################
#######################################
#######################################
#######################################
#######################################
#######################################
#######################################

# traj tracking stuff
def enu_ned(v):
    return np.array([v[1], v[0], -v[2]])            # (E,N,U) <-> (N,E,D), self-inverse

def request_fast_state(m, hz=25):
    """Bump LOCAL_POSITION_NED to control-loop rate (default streams are too slow)."""
    m.mav.command_long_send(
        m.target_system, m.target_component,
        M.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        M.MAVLINK_MSG_ID_LOCAL_POSITION_NED, int(1e6 / hz), 0, 0, 0, 0, 0)

def get_state_enu(m):
    msg = m.recv_match(type='LOCAL_POSITION_NED', blocking=True, timeout=1)
    if msg is None:
        return None
    p_enu = np.array([msg.y, msg.x, -msg.z])
    v_enu = np.array([msg.vy, msg.vx, -msg.vz])
    return np.concatenate([p_enu, v_enu])

def send_accel(m, a_ned):
    m.mav.set_position_target_local_ned_send(
        0, m.target_system, m.target_component,
        M.MAV_FRAME_LOCAL_NED, ACCEL_ONLY,
        0, 0, 0,  0, 0, 0,
        a_ned[0], a_ned[1], a_ned[2],
        0, 0)

def control_law(x, t, controller, ref):
    p_ref, v_ref = ref(t)
    return controller.compute_u(x, p_ref, v_ref)

def fly_trajectory(m, ref, controller, duration, dt=0.04):
    t0 = time.time()
    while (t := time.time() - t0) <= duration:
        x = get_state_enu(m)
        if x is None:
            continue
        u_enu = control_law(x, t, controller, ref)
        send_accel(m, enu_ned(u_enu))
        time.sleep(dt)
    print("trajectory done")


if __name__ == '__main__':
    connection = "udp:127.0.0.1:14550"
    takeoff_altitude = 10

    m = connect(connection)
    wait_until_armable(m)
    set_mode(m, "GUIDED")
    arm(m)
    takeoff(m, takeoff_altitude)

    request_fast_state(m, hz=25)

    print("1 m/s^2 up for 3 s")
    t0 = time.time()
    while time.time() - t0 < 3:
        send_accel(m, np.array([0,0,-1])) # NED -Z = up
        time.sleep(0.04)


    land(m)
