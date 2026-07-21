import sim.drone_test_only_mission_manager as drone_only_mission
import sim.payload_mission_manager as payload_control_mission
import sim.config as config

from comms.control import get_state_enu

from logs.flight import FlightLogger
import time
import numpy as np

class SafeTrajectory:
    """
    ENU
    Generates and maintains information for a usable/safe reference trajectory for hardware and SITL runs.
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
            safety = self._evaluate_starting_safety(m, p_start, p_end, speed, startPointHoverTime, endPointHoverTime, logger)

            if safety:
                print("Starting location is safe")
                self.p0 = p_start
            if safety == False:
                raise RuntimeError("Unsafe starting conditions, aborting autonomy test")

        # check if we also want the end relative to wherever we started
        if relativeEnd:
            self.p1 = self.p0 + p_end
        else:
            self.p1 = p_end

        # initialize mission parameters
        self.speed = speed
        self.startPointHoverTime = startPointHoverTime
        self.endPointHoverTime = endPointHoverTime


    def drone_trajectory(self):
        """
        Computes the reference trajectory object for only the drone
        """
        reference_trajectory_obj = drone_only_mission.ReferenceTrajectory(self.p0, self.p1, self.speed,
                                                                           self.startPointHoverTime, self.endPointHoverTime)
        return reference_trajectory_obj


    def payload_trajectory(self):
        """
        Computes the reference trajectory object for the payload
        """
        # TODO: implement this section when payload reference trajectory is ready
        raise NotImplementedError("Payload trajectory generation will be added soon")





    @staticmethod
    def _evaluate_starting_safety(m, p_start, p_end, speed, startPointHoverTime, endPointHoverTime, logger) -> bool:
        """
        Evaluates the safety of the requested initial conditions
        """
        safe_starting_delta = 1 # acceptable starting deviation
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
        delta = np.linalg.norm(p_start - current_position)

        # unsafe position notification
        if delta > safe_starting_delta:
            print("DANGER")
            print(f"Difference between requested start position and current start position is: {delta} m")
            user_response_to_unsafe = input("Do you wish to continue? (YES/NO): ")

            # check user response
            if user_response_to_unsafe == "NO":
                raise RuntimeError("Unsafe starting position, aborting autonomy test")

            if user_response_to_unsafe == "YES":
                safe_start_pos_flag = True

        # safe position notification
        if delta <= safe_starting_delta:
            safe_start_pos_flag = True

        return safe_start_pos_flag
