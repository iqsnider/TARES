from pymavlink import mavutil
import time
import numpy as np

import Prm.config as config
import sim.dynamics as dynamics
import sim.estimation.ekf as ekfm

from logs.flight import FlightLogger

import comms.estimator as est


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


class ControlComms:
    """
    A library of functions for sending/managing control communications for a given mav connection.
    Makes sure the time and logger are continuosly operating from object intialization.
    """

    def __init__(self, m, control_frequency=50, logger=None):
        # connection and frequency availability
        self.m = m
        self.hz = control_frequency
        self.x0 = None

        # flight-wide clock zero, set once on the first fly_*_trajectory call
        # so a multi-leg flight plan logs one continuous clock, not a reset
        # timer per leg
        self._t_flight0 = None

        # set once a control loop sees the pilot take the aircraft back;
        # defined here so cleanup can read it even if no loop ever ran
        self.pilot_override = False

        # tell mavlink to stream data faster
        self._request_fast_state()

        # logger initialization
        # sent_mode is left alone here: it means the last mode this script
        # commanded, not whatever the FC happened to be in at startup
        if logger == None:
            self.logger = FlightLogger()
        else:
            self.logger = logger

        # initialize control states
        self._initialize_control_logs()

    def _initialize_control_logs(self):
        """
        Gets initial state for control logic
        """
        # wait for attitude and NED
        t_wait = time.time() + 5
        while self.x0 is None or np.isnan(self.logger.cache['yaw']):
            # get values from fc
            self.logger.pump(self.m)

            # intialize the state
            self.x0 = self.get_state_enu(self.logger.cache['ned'], prev=None)

            if time.time() > t_wait:
                raise RuntimeError(
                    "no LOCAL_POSITION_NED / ATTITUDE within 5s")

            # wait 10ms before retrying
            time.sleep(0.01)


    def _wait_fresh_state(self):
        """
        Pump until a new LOCAL_POSITION_NED arrives, so the first control tick
        isn't computed from a state that went stale during setup.
        """
        stale = self.logger.cache['fc_time_boot_ms']
        while self.logger.cache['fc_time_boot_ms'] == stale:
            self.logger.pump(self.m)
            time.sleep(0.002)

    @staticmethod
    def enu_ned(v):
        """
        (E,N,U) <-> (N,E,D), self-inverse
        """
        return np.array([v[1], v[0], -v[2]])

    def set_rate(self, name, hz):
        """
        Set one message's stream interval via SET_MESSAGE_INTERVAL (cmd 511).
        hz <= 0 disables the message
        """
        msg_id = getattr(M, f"MAVLINK_MSG_ID_{name}")
        interval_us = -1 if hz <= 0 else int(1e6 / hz)   # -1 = disable
        self.m.mav.command_long_send(
            self.m.target_system, self.m.target_component,
            M.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            msg_id, interval_us, 0, 0, 0, 0, 0)

    def _request_fast_state(self):
        """
        Bump LOCAL_POSITION_NED to control-loop rate (default streams are too slow).
        """
        rates = {
            "LOCAL_POSITION_NED": self.hz,  # controller state -> control rate
            "ATTITUDE": self.hz,
            "POSITION_TARGET_LOCAL_NED": self.hz,
            "RAW_IMU": self.hz,
            "RC_CHANNELS": self.hz,
            "EKF_STATUS_REPORT": 5,
            "GPS_RAW_INT": 2,
            "SYS_STATUS":  2,  # battery
        }
        for name, rate in rates.items():
            self.set_rate(name, rate)
            time.sleep(0.05)

    @staticmethod
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

    def send_accel(self, a_ned, yaw=None):
        """
        sends the acceleration setpoint to ardupilot and returns the bitmask
        """
        if yaw is not None:
            mask = ACCEL_ONLY_LOCK_YAW
            self.m.mav.set_position_target_local_ned_send(
                0, self.m.target_system, self.m.target_component,
                M.MAV_FRAME_LOCAL_NED, mask,
                0, 0, 0,  0, 0, 0,
                a_ned[0], a_ned[1], a_ned[2],
                yaw, 0)
        else:
            mask = ACCEL_ONLY
            self.m.mav.set_position_target_local_ned_send(
                0, self.m.target_system, self.m.target_component,
                M.MAV_FRAME_LOCAL_NED, mask,
                0, 0, 0,  0, 0, 0,
                a_ned[0], a_ned[1], a_ned[2],
                0, 0)
        return mask

    def _debug_stream_rate(self, t, state):
        """
        Report LOCAL_POSITION_NED arrival rate and re-assert the fast streams.

        state is the mutable dict the caller keeps across iterations.
        """
        lp = self.logger.cache['fc_time_boot_ms']
        if lp != state['last_lp']:
            state['n_lp'] += 1
            state['last_lp'] = lp

        if t - state['last_report'] > 1:
            print(f"LOCAL_POSITION_NED ~{state['n_lp']/(t-state['last_report']):.1f} Hz")
            state['n_lp'] = 0
            state['last_report'] = t

        if t - state['last_reassert'] > 1:
            self.set_rate("LOCAL_POSITION_NED", 50)
            self.set_rate("ATTITUDE", 50)
            self.set_rate("RAW_IMU", 50)
            state['last_reassert'] = t

    def fly_drone_trajectory(self, ref, controller, duration, yaw_lock=True, yaw_ref=None, reassert=False):
        """
        Trajectory following control loop for only the drone
        """

        # compute time step
        dt = 1/self.hz

        # get initial state
        self._wait_fresh_state()
        x = self.x0

        # set yaw_ref to current yaw when yaw locking and no yaw reference is specified
        if yaw_lock and yaw_ref is None:
            yaw_ref = self.logger.cache['yaw']

        # intiialize debugging values
        dbg = {'last_reassert': 0,
               'last_lp': self.logger.cache['fc_time_boot_ms'],
               'last_report': 0,
               'n_lp': 0}

        # intialize time for control loop
        t0 = time.time()
        next_t = t0
        if self._t_flight0 is None:
            self._t_flight0 = t0
        leg_offset = t0 - self._t_flight0

        # begin the control loop and run for the duration of the reference trajectory
        while (t := time.time() - t0) <= duration:
            self.logger.pump(self.m)

            if reassert:
                self._debug_stream_rate(t, dbg)

            # get current state p,v
            x = self.get_state_enu(self.logger.cache['ned'], prev=x)

            # get the reference p,v from the trajectory
            p_ref, v_ref = ref(t)

            # compute the control input (acceleration setpoints)
            u = controller.compute_u(x, p_ref, v_ref)

            # set setpoint msg bitmasks depending on whether or not we are commanding yaw
            if yaw_lock:
                mask = self.send_accel(self.enu_ned(u), yaw=yaw_ref)
            else:
                mask = self.send_accel(self.enu_ned(u))

            # confirm the bitmask with the FC
            self.logger.note_sent(bitmask=mask)

            # log sent values, on the flight-wide clock rather than this leg's
            if yaw_ref is not None:
                self.logger.log(t + leg_offset, x, p_ref, v_ref, u, yaw_ref=yaw_ref)
            else:
                self.logger.log(t + leg_offset, x, p_ref, v_ref, u)

            # time step
            next_t += dt
            time.sleep(max(0, next_t - time.time()))

    def fly_payload_trajectory(self, ref, controller, duration, recorder, ekf,
                               yaw_lock=True, yaw_ref=None, reassert=False):
        """
        Closed-loop payload reference tracking
        """
        # compute time step
        dt = 1/self.hz
        L = config.TETHER_LEN

        # get initial state
        self._wait_fresh_state()
        x = self.x0

        # set yaw_ref to current yaw when yaw locking and no yaw reference is specified
        if yaw_lock and yaw_ref is None:
            yaw_ref = self.logger.cache['yaw']

        # intiialize debugging values
        dbg = {'last_reassert': 0,
               'last_lp': self.logger.cache['fc_time_boot_ms'],
               'last_report': 0,
               'n_lp': 0}

        last_seq = -1

        # the loop time the filter last integrated to; None until the first
        # pass, which has no elapsed time to measure and uses the nominal one
        t_prev = None

        # set when the pilot takes the aircraft back, so the caller knows the
        # vehicle is no longer ours to command on the way out
        self.pilot_override = False

        # intialize time for control loop
        t0 = time.time()
        next_t = t0
        if self._t_flight0 is None:
            self._t_flight0 = t0
        leg_offset = t0 - self._t_flight0

        # begin the control loop and run for the duration of the reference trajectory
        while (t := time.time() - t0) <= duration:
            self.logger.pump(self.m)
            c = self.logger.cache

            # the pilot flipping out of GUIDED ends the test
            if c['echoed_mode'] not in ("GUIDED", "?"):
                self.pilot_override = True
                print(f"pilot override: mode is {c['echoed_mode']}, "
                      f"stopping at t={t:.1f}s")
                break

            if reassert:
                self._debug_stream_rate(t, dbg)

            # get current drone state p,v
            x = self.get_state_enu(c['ned'], prev=x)

            # fold in the camera only when the frame is new
            seq, meas = est.latest_measurement(recorder, ekf.source)
            if seq == last_seq:
                meas = None
            else:
                last_seq = seq

            dt_ekf = dt if t_prev is None else t - t_prev
            t_prev = t

            # payload swing estimate: [alpha_x alpha_y alpha_dot_x alpha_dot_y psi_p]
            xi, P = est.step_ekf(ekf, meas, est.accel_enu(c), dt_ekf,
                                 c['roll'], c['pitch'], c['yaw'])

            # payload reference, lifted to the drone equilibrium
            p_ref, v_ref = ref(t)
            x_ref = dynamics.tether_equilibrium_state(p_ref, v_ref, L)

            # assemble the measured 16-state and compute the control input
            x16 = est.payload_state_16(x, xi)
            u = controller.compute_u(x16 - x_ref)

            # set setpoint msg bitmasks depending on whether or not we are commanding yaw
            if yaw_lock:
                mask = self.send_accel(self.enu_ned(u), yaw=yaw_ref)
            else:
                mask = self.send_accel(self.enu_ned(u))

            # confirm the bitmask with the FC
            self.logger.note_sent(bitmask=mask)

            # log sent values, on the flight-wide clock rather than this leg's.
            # drone_*_ref is the lifted equilibrium the drone is actually
            # chasing; payload_*_ref is what the mission asked for.
            self.logger.log(t + leg_offset, x, u=u,
                            yaw_ref=yaw_ref if yaw_ref is not None else 0,
                            payload_p_ref=p_ref,
                            payload_v_ref=v_ref,
                            payload_alpha=(xi[0], xi[1]),
                            payload_alphadot=(xi[2], xi[3]),
                            payload_psi_p=xi[ekfm.IX_PSI_P],
                            payload_innov=ekf.innov,
                            payload_cov=(P[ekfm.IX_ALPHA_X, ekfm.IX_ALPHA_X],
                                        P[ekfm.IX_ALPHA_Y, ekfm.IX_ALPHA_Y],
                                        P[ekfm.IX_ALPHA_X, ekfm.IX_ALPHA_Y],
                                        P[ekfm.IX_PSI_P, ekfm.IX_PSI_P]))

            # time step
            next_t += dt
            time.sleep(max(0, next_t - time.time()))


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


# check math
if __name__ == '__main__':
    print(ACCEL_ONLY_LOCK_YAW)
