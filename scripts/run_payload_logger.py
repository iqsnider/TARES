"""
Passive logger: records the drone state and the payload swing estimate with
nothing commanded, so a manually flown flight can be studied afterwards.
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
    connection = "/dev/ttyACM0"
    baud = 115200
    # connection = "udp:127.0.0.1:14550"
    control_freq = config.CONTROL_FREQUENCY

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/test_09022026/payload_log_{stamp}"
    video_out = f"{data_dir}/recording.avi"

    if config.EKF_SOURCE == "aruco":
        poses_out = f"{data_dir}/poses.csv"
    else:
        poses_out = f"{data_dir}/circles.csv"

    # add baud here if connected to real drone
    m = comms.connect(connection, baud)

    # intialize the logs
    logger = FlightLogger(data_dir=data_dir)

    # start camera first
    recorder, cam_thread = cam.start_payload_camera(video_out=video_out,
                                                    csv_out=poses_out)

    try:
        # no control is sent, this is only here for the fast state streams the
        # filter needs and the first state the loop starts from
        controlLink = ControlComms(m,
                                   control_frequency=control_freq,
                                   logger=logger)

        # payload swing estimator: attitude comes from the logger cache, which
        # ControlComms has already populated by blocking for the first state
        ekf = est.start_ekf(logger, recorder=recorder)

        # log until ctrl-c
        print("logging payload swing, ctrl-c to stop...")
        controlLink.watch_payload(recorder, ekf)
    except KeyboardInterrupt:
        print("stopping")
    finally:
        # stop the camera first so it flushes its video and pose csv
        # then close the flight logger
        recorder.stop()
        cam_thread.join(timeout=5)
        logger.close()
