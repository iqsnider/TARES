from pymavlink import mavutil
import time
import numpy as np

import sim.config as config
import sim.drone_test_only_mission_manager as mission
import sim.SITL_dynamics as dynamics

from logs.flight import FlightLogger

M = mavutil.mavlink

ACCEL_ONLY = (M.POSITION_TARGET_TYPEMASK_X_IGNORE
              | M.POSITION_TARGET_TYPEMASK_Y_IGNORE
              | M.POSITION_TARGET_TYPEMASK_Z_IGNORE
              | M.POSITION_TARGET_TYPEMASK_VX_IGNORE
              | M.POSITION_TARGET_TYPEMASK_VY_IGNORE
              | M.POSITION_TARGET_TYPEMASK_VZ_IGNORE
              | M.POSITION_TARGET_TYPEMASK_YAW_IGNORE
              | M.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)

ACCEL_ONLY_LOCK_YAW = (M.POSITION_TARGET_TYPEMASK_X_IGNORE
                       | M.POSITION_TARGET_TYPEMASK_Y_IGNORE
                       | M.POSITION_TARGET_TYPEMASK_Z_IGNORE
                       | M.POSITION_TARGET_TYPEMASK_VX_IGNORE
                       | M.POSITION_TARGET_TYPEMASK_VY_IGNORE
                       | M.POSITION_TARGET_TYPEMASK_VZ_IGNORE)


def enu_ned(v):
    """
    (E,N,U) <-> (N,E,D), self-inverse
    """
    return np.array([v[1], v[0], -v[2]])


def set_rate(m, name, hz):
    """
    Set one message's stream interval via SET_MESSAGE_INTERVAL (cmd 511).
    hz <= 0 disables the message
    """
    msg_id = getattr(M, f"MAVLINK_MSG_ID_{name}")
    interval_us = -1 if hz <= 0 else int(1e6 / hz)   # -1 = disable
    m.mav.command_long_send(
        m.target_system, m.target_component,
        M.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        msg_id, interval_us, 0, 0, 0, 0, 0)


def request_fast_state(m, hz=50):
    """
    Bump LOCAL_POSITION_NED to control-loop rate (default streams are too slow).
    """
    rates = {
        "LOCAL_POSITION_NED": hz,  # controller state -> control rate
        "ATTITUDE": hz,
        "POSITION_TARGET_LOCAL_NED": hz,
        "RC_CHANNELS": 10,
        "EKF_STATUS_REPORT": 5,
        "GPS_RAW_INT": 2,
        "SYS_STATUS":  2,  # battery
    }
    for name, rate in rates.items():
        set_rate(m, name, rate)
        time.sleep(0.05)


def get_state_enu(ned, prev=None):
    """
    Converts NED state to ENU state
    """
    if ned is None or np.isnan(ned[0]):
        return prev  # nothing yet this session -> reuse last known state
    x, y, z, vx, vy, vz = ned
    p_enu = np.array([y, x, -z])
    v_enu = np.array([vy, vx, -vz])

    return np.concatenate([p_enu, v_enu])


def send_accel(m, a_ned, yaw=None):
    """
    sends the acceleration setpoint to ardupilot and returns the bitmask
    """
    if yaw is not None:
        mask = ACCEL_ONLY_LOCK_YAW
        m.mav.set_position_target_local_ned_send(
            0, m.target_system, m.target_component,
            M.MAV_FRAME_LOCAL_NED, mask,
            0, 0, 0,  0, 0, 0,
            a_ned[0], a_ned[1], a_ned[2],
            yaw, 0)
    else:
        mask = ACCEL_ONLY
        m.mav.set_position_target_local_ned_send(
            0, m.target_system, m.target_component,
            M.MAV_FRAME_LOCAL_NED, mask,
            0, 0, 0,  0, 0, 0,
            a_ned[0], a_ned[1], a_ned[2],
            0, 0)
    return mask



def fly_trajectory(m, ref, controller, duration, dt, yaw_lock=False, reassert=False):
    logger = FlightLogger()
    logger.note_sent(mode=m.flightmode)

    # block for the first real state before commanding anything
    x = None
    t_wait = time.time() + 5
    while x is None:
        logger.pump(m)
        x = get_state_enu(logger.cache['ned'], prev=None)
        if time.time() > t_wait:
            raise RuntimeError("no LOCAL_POSITION_NED within 5s")
        time.sleep(0.01)

    t0 = time.time()
    next_t = t0

    # debugging
    if reassert:
        last_reassert = 0
        last_lp = logger.cache['fc_time_boot_ms']
        last_report = 0
        n_lp = 0

    while (t := time.time() - t0) <= duration:
        logger.pump(m)

        # debugging
        if reassert:
            # checking frequency
            lp = logger.cache['fc_time_boot_ms']
            if lp != last_lp:
                n_lp += 1
                last_lp = lp

            if t - last_report > 1:
                print(f"LOCAL_POSITION_NED ~{n_lp/(t-last_report):.1f} Hz")
                n_lp = 0
                last_report = t

            # re-assert the critical fast stream 1 Hz
            if t - last_reassert > 1:
                set_rate(m, "LOCAL_POSITION_NED", 50)
                last_reassert = t

        x = get_state_enu(logger.cache['ned'], prev=x)
        p_ref, v_ref = ref(t)
        u = controller.compute_u(x, p_ref, v_ref)
        mask = send_accel(m, enu_ned(u))
        logger.note_sent(bitmask=mask)
        logger.log(t, x, p_ref, v_ref, u)
        next_t += dt
        time.sleep(max(0, next_t - time.time()))

    logger.close()


def fly_trajectory_goldfish(m, ref, controller, duration, dt, yaw_lock=False):
    t0 = time.time()
    next_t = t0
    x = None
    while (t := time.time() - t0) <= duration:
        x = get_state_enu(m, prev=x)

        p_ref, v_ref = ref(t)
        u = controller.compute_u(x, p_ref, v_ref)
        send_accel(m, enu_ned(u))
        next_t += dt
        time.sleep(max(0, next_t - time.time()))


if __name__ == '__main__':
    print(ACCEL_ONLY)
