import sim.mission_manager as mission
import Prm.config as config

from comms.control import get_state_enu

from logs.flight import FlightLogger
import time
import numpy as np


class SafeFlightPlan:
    """
    Generates a flight plan from a given waypoint file or dictionary, in which each step in the dictionary moves ENU from the current position in meters.
    Creates SafeTrajectory objects between each waypoint.
    """

    def __init__(self,
                 m,  # MAVLink connection
                 wp_file=None,  # TODO: waypoint file parser
                 wp_dict={1: [-20, 0, 0],
                          2: [0, -20, 0],
                          3: [20, 0, 0],
                          4: [0, 20, 0]},
                 wp_hover_time=15,  # [s]
                 speed=1,  # [m/s]
                 logger=None):

        if wp_file is not None:
            raise NotImplementedError(
                "waypoint file parsing is not yet implemented, pass wp_dict instead")

        # initialization
        if logger == None:
            self.logger = FlightLogger()
            self.logger.note_sent(mode=m.flightmode)
        else:
            self.logger = logger
            self.logger.note_sent(mode=m.flightmode)

        self.m = m
        self.wp_dict = wp_dict
        self.wp_hover_time = wp_hover_time
        self.speed = speed

    def legs(self):
        """
        builds a SafeTrajectory for each waypoint in wp_dict, in key
        order. Each waypoint's displacement is relative ENU meters.
        """
        for key in sorted(self.wp_dict):
            p_end = self.wp_dict[key]
            yield SafeTrajectory(self.m, None, p_end, self.speed,
                                 startPointHoverTime=self.wp_hover_time,
                                 endPointHoverTime=self.wp_hover_time,
                                 startFromCurrentPosition=True,
                                 relativeEnd=True,
                                 logger=self.logger)

    def drone_trajectories(self):
        """
        Yields the drone-only reference trajectory for each leg of the plan, in order.
        """
        for leg in self.legs():
            yield leg.drone_trajectory()

    def payload_trajectories(self):
        """
        Yields the payload reference trajectory for each leg of the plan, in order.
        """
        for leg in self.legs():
            yield leg.payload_trajectory()


class SafeTrajectory:
    """
    ENU
    Generates and maintains information for a usable/safe reference trajectory for hardware and SITL runs.

    p_start and p_end are DRONE positions. payload_trajectory() offsets them
    down by one tether length to produce the payload reference.
    """

    def __init__(self, m, p_start, p_end, speed, startPointHoverTime=1, endPointHoverTime=1, startFromCurrentPosition=True, relativeEnd=True, logger=None):
        # initialization
        if logger == None:
            self.logger = FlightLogger()
            self.logger.note_sent(mode=m.flightmode)
        else:
            self.logger = logger
            self.logger.note_sent(mode=m.flightmode)

        # check if start condition request is for current position and find initial position
        if startFromCurrentPosition:
            print("Starting from current position, initializing trajectory...")
            # block for the first real state
            x0 = None
            t_wait = time.time() + 5
            while x0 is None:
                self.logger.pump(m)
                x0 = get_state_enu(self.logger.cache['ned'], prev=None)
                if time.time() > t_wait:
                    raise RuntimeError("no LOCAL_POSITION_NED within 5s")
                time.sleep(0.01)

            self.p0 = x0[:3]

        # if not start from current position, evaluate safety of requested starting position
        else:
            print("Starting from predefined location, evaluating safety...")
            safety = self._evaluate_starting_safety(
                m, p_start, p_end, speed, startPointHoverTime, endPointHoverTime, self.logger)

            if safety:
                print("Starting location is safe")
                self.p0 = np.asarray(p_start, dtype=float)
            if safety == False:
                raise RuntimeError(
                    "Unsafe starting conditions, aborting autonomy test")

        # check if we also want the end relative to wherever we started
        if relativeEnd:
            self.p1 = self.p0 + p_end
        else:
            self.p1 = np.asarray(p_end, dtype=float)

        # initialize mission parameters
        self.speed = speed
        self.startPointHoverTime = startPointHoverTime
        self.endPointHoverTime = endPointHoverTime

    def drone_trajectory(self):
        """
        Computes the reference trajectory object for only the drone
        """
        reference_trajectory_obj = mission.ReferenceTrajectory(self.p0, self.p1, self.speed,
                                                               self.startPointHoverTime, self.endPointHoverTime)
        return reference_trajectory_obj

    def payload_trajectory(self):
        """
        Computes the reference trajectory object for the payload
        """
        tether = np.array([0, 0, config.TETHER_LEN])
        reference_trajectory_obj = mission.ReferenceTrajectory(self.p0 - tether, self.p1 - tether, self.speed,
                                                               self.startPointHoverTime, self.endPointHoverTime)
        return reference_trajectory_obj

    @staticmethod
    def _evaluate_starting_safety(m, p_start, p_end, speed, startPointHoverTime, endPointHoverTime, logger) -> bool:
        """
        Evaluates the safety of the requested initial conditions
        """
        safe_starting_delta = 1  # acceptable starting deviation
        safe_start_pos_flag = False

        # request current position
        current_position = None
        t_wait = time.time() + 5
        while current_position is None:
            logger.pump(m)
            current_position = get_state_enu(logger.cache['ned'], prev=None)
            if time.time() > t_wait:
                raise RuntimeError("no LOCAL_POSITION_NED within 5s")
            time.sleep(0.01)

        # evaluate difference between p_start and current_position
        # get_state_enu returns [p, v]; only the position half is comparable
        delta = np.linalg.norm(np.asarray(
            p_start, dtype=float) - current_position[:3])

        # unsafe position notification
        if delta > safe_starting_delta:
            print("DANGER")
            print(f"Difference between requested start position and current start position is: {
                  delta} m")
            user_response_to_unsafe = input(
                "Do you wish to continue? (YES/NO): ")

            # check user response
            if user_response_to_unsafe == "NO":
                raise RuntimeError(
                    "Unsafe starting position, aborting autonomy test")

            if user_response_to_unsafe == "YES":
                safe_start_pos_flag = True

        # safe position notification
        if delta <= safe_starting_delta:
            safe_start_pos_flag = True

        return safe_start_pos_flag
