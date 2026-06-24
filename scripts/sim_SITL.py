from pymavlink import mavutil
import time

import sim.config as config
import sim.simplified_mission_manager as mission
import sim.SITL_dynamics as dynamics


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
        time.sleep()

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


def control_law(x, t, outerLoopControl, innerLoopControl, ref):
    """Flight controller about the trajectory equilibrium at time t"""
    p_PL_ref, v_PL_ref = ref(t)  # current reference

    x_star = mission.equilibrium_state(
        p_PL_ref, v_PL_ref, config.TETHER_LEN)  # equilibrium state

    e = x - x_star  # error

    e[6:9] = dynamics.wrap_angle(e[6:9])  # angle wrapping

    # Outer-loop LQR controller
    a_des = outerLoopControl.compute_u(e)

    # Inner-loop ardupilot flight controller [C_Sigma, n1, n2, n3]
    u_pert = innerLoopControl.compute_u(x, a_des, yaw_s=0)
    u = u_pert

    return u, u_pert, e


if __name__ == '__main__':
    connection = "udp:127.0.0.1:14540"
    takeoff_altitude = 10
    hover_time = 20

    m = connect(connection)
    wait_until_armable(m)
    set_mode(m, "GUIDED")
    arm(m)
    takeoff(m, takeoff_altitude)
    hover(m, hover_time)
    land(m)
