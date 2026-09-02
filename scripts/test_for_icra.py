"""
Control (experimental control as in the baseline to compare against control, not control as in "control theory") test for ICRA data submission.

Flies a square on ardupilot's own position controller with the logger and the
payload EKF running, so the swing is recorded but never fed back.
"""
# mission control imports
import comms.common as comms
from comms.control import ControlComms

# autonomy research imports
import comms.camera as cam
import comms.estimator as est
import Prm.config as config

# logging
from logs.flight import FlightLogger


from datetime import datetime


if __name__ == '__main__':
    # connection = "/dev/ttyACM0"
    # baud = 115200
    connection = "udp:127.0.0.1:14550"
    takeoff_altitude = 15
    control_freq = config.CONTROL_FREQUENCY
    speed = 1

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/pretest_09012026/ardupilot_survey_test_{stamp}"
    video_out = f"{data_dir}/recording.avi"

    if config.EKF_SOURCE == "aruco":
        poses_out = f"{data_dir}/poses.csv"
    else:
        poses_out = f"{data_dir}/circles.csv"

    # add baud here if connected to real drone
    m = comms.connect(connection)

    # check for armability
    comms.wait_until_armable(m)

    # GUIDED mode is easiest for external commands
    comms.set_guid_options(m, 0)
    comms.set_mode(m, "GUIDED")

    # arm the motors if not already armed
    comms.arm(m)
    #
    # # CLEAR THE AREA
    comms.takeoff(m, takeoff_altitude)

    # intialize the logs
    logger = FlightLogger(data_dir=data_dir)

    # start camera first
    recorder, cam_thread = cam.start_payload_camera(video_out=video_out,
                                                    csv_out=poses_out)

    controlLink = None
    try:
        # initalize control communications and prepare datastream for high rate control requests
        controlLink = ControlComms(m,
                                   control_frequency=control_freq,
                                   logger=logger)

        # mission reference
        edge_length = 20
        wp_hover_time = 20

        # payload swing estimator: attitude comes from the logger cache, which
        # ControlComms has already populated by blocking for the first state.
        # Watching only on this run, nothing closes around it
        ekf = est.start_ekf(logger, recorder=recorder)

        # run the baseline: ardupilot flies every side of the square
        print("running ardupilot baseline...")
        controlLink.fly_ardupilot_square(edge_length, wp_hover_time, speed,
                                         recorder=recorder, ekf=ekf)
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
            recorder.stop()
            cam_thread.join(timeout=5)
            logger.close()
