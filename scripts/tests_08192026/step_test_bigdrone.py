"""
Big drone
"""
# mission control imports
import comms.common as comms
from comms.control import ControlComms

# autonomy research imports
import comms.trajectory as mission
import sim.dynamics as dynamics
import comms.camera as cam
import comms.estimator as est
import Prm.config as config

# logging
from logs.flight import FlightLogger


from datetime import datetime


if __name__ == '__main__':
    connection = "/dev/ttyACM0"
    baud = 115200
    # connection = "udp:127.0.0.1:14550"
    # takeoff_altitude = 15
    control_freq = config.CONTROL_FREQUENCY
    speed = 1

    # marker / camera-output config
    track_marker_ids = [config.LEFT_MARKER_ID, config.CENTER_MARKER_ID, config.RIGHT_MARKER_ID]   # only these are logged; others dropped
    preview_port = None                  # live MJPEG view; None to disable

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/test_08192026/step_test_bigdrone_{stamp}"
    video_out = f"{data_dir}/recording.avi"
    poses_out = f"{data_dir}/poses.csv"

    # add baud here if connected to real drone
    m = comms.connect(connection, baud)

    # check for armability
    # comms.wait_until_armable(m)


    # GUIDED mode is easiest for external commands
    comms.set_guid_options(m, 0)
    comms.set_mode(m, "GUIDED")

    # arm the motors if not already armed
    # comms.arm(m)
    #
    # # CLEAR THE AREA
    # comms.takeoff(m, takeoff_altitude)

    # intialize the logs
    logger = FlightLogger(data_dir=data_dir)


    controlLink = None
    try:
        # initalize control communications and prepare datastream for high rate control requests
        controlLink = ControlComms(m,
                                   control_frequency=control_freq,
                                   logger=logger)

        # mission reference
        startPointHoverTime = 30
        endPointHoverTime = 30

        # ENU to ENU. payload_trajectory()
        # reference is where payload should be
        ref = mission.SafeTrajectory(m, None, [0, -20, 0], speed=speed,
                                     startPointHoverTime=startPointHoverTime,
                                     endPointHoverTime=endPointHoverTime,
                                     startFromCurrentPosition=True,
                                     relativeEnd=True,
                                     logger=logger).drone_trajectory()

        # define outer-loop control law
        controller = dynamics.OuterLoopLQR()
        logger.set_controller(controller)


        # run autonomy
        print("running bigdrone LQR controller...")

        # tell ardupilot not to help the external control system with the GUID_OPTION mode of 48
        comms.set_guid_options(m, 48)

        controlLink.fly_drone_trajectory(ref,
                                           controller,
                                           duration=ref.duration,
                                           yaw_lock=True,
                                           reassert=False)
    finally:
        try:
            if controlLink is None or not controlLink.pilot_override:
                comms.set_mode(m, "BRAKE")
        finally:
            # flush whatever happened above: a mode change that times out
            # must not cost us the recording as well.
            # stop the camera first so it flushes its video and pose csv
            # then close the flight logger
            comms.set_guid_options(m, 0)
            logger.close()
