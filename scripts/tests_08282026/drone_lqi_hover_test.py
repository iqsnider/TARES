"""
drone lqi hover test
"""
# mission control imports
import comms.common as comms
from comms.control import ControlComms

# autonomy research imports
import sim.dynamics as dynamics
import comms.camera as cam
import comms.estimator as est
import Prm.config as config

# logging
from logs.flight import FlightLogger

# math imports
import numpy as np

from datetime import datetime


HOVER_S = 60


if __name__ == '__main__':
    connection = "/dev/ttyACM0"
    baud = 115200
    control_freq = config.CONTROL_FREQUENCY

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/test_08282026/drone_lqi_hover_test_{stamp}"
    video_out = f"{data_dir}/recording.avi"

    if config.EKF_SOURCE == "aruco":
        poses_out = f"{data_dir}/poses.csv"
    else:
        poses_out = f"{data_dir}/circles.csv"

    # add baud here if connected to real drone
    m = comms.connect(connection, baud)

    # GUIDED mode is easiest for external commands
    comms.set_guid_options(m, 0)
    comms.set_mode(m, "GUIDED")

    # intialize the logs
    logger = FlightLogger(data_dir=data_dir)

    controlLink = None
    recorder = cam_thread = ekf = None
    try:
        # initalize control communications and prepare datastream for high rate control requests
        controlLink = ControlComms(m,
                                   control_frequency=control_freq,
                                   logger=logger)

        try:
            recorder, cam_thread = cam.start_payload_camera(
                video_out=video_out,
                csv_out=poses_out)
            ekf = est.start_ekf(logger, recorder=recorder)
        except Exception as e:
            print(f"payload filter not running ({e}); hovering without it")
            ekf = None

        p_hover = controlLink.x0[0:3].copy()

        def ref(t):
            return p_hover, np.zeros(3)

        # define outer-loop control law
        controller = dynamics.OuterLoopLQI()
        logger.set_controller(controller)

        # run autonomy
        print(f"holding {p_hover.round(2)} ENU for {HOVER_S}s with LQI...")

        # tell ardupilot not to help the external control system with the GUID_OPTION mode of 48
        comms.set_guid_options(m, 48)

        controlLink.fly_drone_trajectory(ref,
                                         controller,
                                         duration=HOVER_S,
                                         yaw_lock=True,
                                         reassert=False,
                                         recorder=recorder,
                                         ekf=ekf)
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
            if recorder is not None:
                recorder.stop()
                cam_thread.join(timeout=5)
            logger.close()
