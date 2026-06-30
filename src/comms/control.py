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


def enu_ned(v):
    """
    (E,N,U) <-> (N,E,D), self-inverse
    """
    return np.array([v[1], v[0], -v[2]])


def request_fast_state(m, hz=50):
    """
    Bump LOCAL_POSITION_NED to control-loop rate (default streams are too slow).
    """
    m.mav.command_long_send(
        m.target_system, m.target_component,
        M.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
        M.MAVLINK_MSG_ID_LOCAL_POSITION_NED, int(1e6 / hz), 0, 0, 0, 0, 0)


def get_state_enu(m, prev=None):
    msg = None
    while (latest := m.recv_match(type='LOCAL_POSITION_NED', blocking=False)) is not None:
        msg = latest
    if msg is None:
        return prev  # nothing new this tick -> reuse last known state
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


def drain_aux(m, cache):
    """Non-blocking: refresh mode/armed/attitude/battery from whatever's queued."""
    while (msg := m.recv_match(
            type=['HEARTBEAT', 'ATTITUDE', 'SYS_STATUS'], blocking=False)) is not None:
        t = msg.get_type()
        if t == 'HEARTBEAT':
            cache['mode'] = mavutil.mode_string_v10(msg)
            cache['armed'] = bool(msg.base_mode & M.MAV_MODE_FLAG_SAFETY_ARMED)
        elif t == 'ATTITUDE':
            cache['rpy'] = (msg.roll, msg.pitch, msg.yaw)
        elif t == 'SYS_STATUS':
            cache['batt_v'] = msg.voltage_battery / 1000.0  # mV -> V
    return cache


def fly_trajectory(m, ref, controller, duration, dt=0.04, log_dir="data"):
    logger = FlightLogger(log_dir)
    cache = {'mode': '?', 'armed': False, 'rpy': (0, 0, 0), 'batt_v': 0.0}
    t0 = time.time()
    next_t = t0
    x = None
    try:
        while (t := time.time() - t0) <= duration:
            x = get_state_enu(m, prev=x)
            if x is None:
                time.sleep(0.005)
                continue
            drain_aux(m, cache)

            p_ref, v_ref = ref(t)
            # named, so we can log it
            u = controller.compute_u(x, p_ref, v_ref)
            send_accel(m, enu_ned(u))

            logger.log(t, cache['mode'], cache['armed'],
                       p_ref, v_ref, x, u, cache['rpy'], cache['batt_v'])

            next_t += dt
            time.sleep(max(0, next_t - time.time()))
    finally:
        logger.close()
        print(f"trajectory done -> {logger.path}")
