from pymavlink import mavutil
import time
import numpy as np

import Prm.config as config
import sim.drone_test_only_mission_manager as mission
import sim.dynamics as dynamics

from logs.flight import FlightLogger
from payload_tracking.aruco_lib import MarkerPoseRecorder


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

        # tell mavlink to stream data faster
        self._request_fast_state()

        # logger initialization
        if logger == None:
            self.logger = FlightLogger()
            self.logger.note_sent(mode=self.m.flightmode)
        else:
            self.logger = logger
            self.logger.note_sent(mode=self.m.flightmode)

        # initialize control states
        self._initialize_control_logs()



    def _initialize_control_logs(self):
        """
        Gets initial state for control logic
        """
        self.logger.note_sent(mode=self.m.flightmode)

        # block for the first real state before commanding anything
        t_wait = time.time() + 5
        while self.x0 is None:
            # get values from fc
            self.logger.pump(self.m)

            # intialize the state
            self.x0 = self.get_state_enu(self.logger.cache['ned'], prev=None)

            if time.time() > t_wait:
                raise RuntimeError("no LOCAL_POSITION_NED within 5s")

            # wait 10ms before retrying
            time.sleep(0.01)


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



    def fly_drone_trajectory(self, ref, controller, duration, yaw_lock=True, yaw_ref=None, reassert=False):
        """
        Trajectory following control loop for only the drone
        """

        # compute time step
        dt = 1/self.hz

        # get initial state
        x = self.x0

        # set yaw_ref to current yaw when yaw locking and no yaw reference is specified
        if yaw_lock and yaw_ref==None:
            yaw_ref = self.logger.cache['yaw']

        # intiialize debugging values
        last_reassert = 0
        last_lp = self.logger.cache['fc_time_boot_ms']
        last_report = 0
        n_lp = 0

        # intialize time for control loop
        t0 = time.time()
        next_t = t0

        # begin the control loop and run for the duration of the reference trajectory
        while (t := time.time() - t0) <= duration:
            self.logger.pump(self.m)

            # debugging
            if reassert:
                # checking frequency
                lp = self.logger.cache['fc_time_boot_ms']
                if lp != last_lp:
                    n_lp += 1
                    last_lp = lp

                if t - last_report > 1:
                    print(f"LOCAL_POSITION_NED ~{n_lp/(t-last_report):.1f} Hz")
                    n_lp = 0
                    last_report = t

                # re-assert the fast stream 1 Hz
                if t - last_reassert > 1:
                    self.set_rate("LOCAL_POSITION_NED", 50)
                    self.set_rate("ATTITUDE", 50)
                    self.set_rate("RAW_IMU", 50)
                    last_reassert = t

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

            # log sent values
            if yaw_ref is not None:
                self.logger.log(t, x, p_ref, v_ref, u, yaw_ref=yaw_ref)
            else:
                self.logger.log(t, x, p_ref, v_ref, u)


            # time step
            next_t += dt
            time.sleep(max(0, next_t - time.time()))


    def get_payload_pose(self):
        """
        Calculates the alpha_x, alpha_y, dalpha_x, and dalpha_y from the aruco pose data
        """
        pass


    def fly_payload_trajectory(self):
        """
        fly the trajectory for payload tracking
        """
        pass



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
