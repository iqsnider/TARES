from pymavlink import mavutil

from payload_tracking.aruco_lib import MarkerPoseRecorder

import comms.common as comms
from comms.control import ControlComms

import time

from logs.flight import FlightLogger

if __name__ == "__main__":
    # connection = "/dev/ttyACM0"
    # baud = 115200
    connection = "udp:127.0.0.1:14550"
    control_freq = 50

    # live view config
    preview_port = 8080                  # MJPEG live view; set None to disable
    track_marker_ids = [219, 220, 221]   # only log these ids; None = all

    # add baud here if connected to real drone
    m = comms.connect(connection)

    # initialize drone logger
    data_dir = "camera_test_data/preflighttest_07222026"
    logger = FlightLogger(data_dir=f"{data_dir}")

    # intializing the contorl object will get the fast state for the logger
    controlLink = ControlComms(m, control_frequency=control_freq, logger=logger, rec=True)

    # begin payload recording
    video_out_data = f"{data_dir}/recording_170mm_test.avi"
    poses_out_data = f"{data_dir}/poses_170mm_test.csv"
    rec = MarkerPoseRecorder(marker_size_m=0.17,
                             video_out=video_out_data,
                             csv_out=poses_out_data,
                             fps=48,
                             marker_ids=track_marker_ids,
                             flight_logger=logger,
                             preview_port=preview_port)
    try:
        rec.run(mav=m)
    finally:
        logger.close()
