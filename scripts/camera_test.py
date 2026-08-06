"""
This test is purely for logging camera data and flight data for a manual flight
"""
from pymavlink import mavutil
import time
from datetime import datetime

from payload_tracking.aruco_lib import MarkerPoseRecorder

import comms.common as comms
import comms.camera as cam
from comms.control import ControlComms

import Prm.config as config


from logs.flight import FlightLogger

if __name__ == "__main__":
    # connection = "/dev/ttyACM0"
    # baud = 115200
    connection = "udp:127.0.0.1:14550"
    control_freq = 50

    # live view config
    preview_port = None# 8080 # live view; set None to disable
    track_marker_ids = [config.LEFT_MARKER_ID, config.CENTER_MARKER_ID, config.RIGHT_MARKER_ID]   # only log these ids; None = all

    # add baud here if connected to real drone
    m = comms.connect(connection)

    # initialize drone logger
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/pretest_08072026/camera_test_{stamp}"
    video_out = f"{data_dir}/recording.avi"
    poses_out = f"{data_dir}/poses.csv"

    logger = FlightLogger(data_dir=f"{data_dir}")

    # intializing the contorl object will get the fast state for the logger
    controlLink = ControlComms(m, control_frequency=control_freq, logger=logger)

    # begin payload recording
    rec = MarkerPoseRecorder(marker_size_m=config.MARKER_EDGE_LEN,
                             video_out=video_out,
                             csv_out=poses_out,
                             capture_fps=48,
                             frame_stride=1,      # 48/2 = 24 fps recorded
                             marker_ids=track_marker_ids,
                             flight_logger=logger,
                             preview_port=preview_port,
                             gain=40,
                             exposure_abs=10)
    try:
        rec.run(mav=m)
    finally:
        logger.close()
