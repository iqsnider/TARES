from pymavlink import mavutil

from payload_tracking.aruco_lib import MarkerPoseRecorder

import comms.common as comms
import comms.control as control

import time

from logs.flight import FlightLogger

if __name__ == "__main__":
    connection = "/dev/ttyACM0"
    baud = 115200
    # connection = "udp:127.0.0.1:14550"
    control_freq = 50

    i = 0
    # add baud here if connected to real drone
    m = comms.connect(connection, baud)
    control.request_fast_state(m, hz=control_freq)

    # initialize drone logger
    logger = FlightLogger(data_dir="camera_test_data")

    # begin payload recording
    video_out_data = f"recording_170mm_test2_iteration_{i}.avi"
    poses_out_data = f"poses_170mm_test2_iteration_{i}.csv"
    rec = MarkerPoseRecorder(marker_size_m=0.17,
                             video_out=video_out_data,
                             csv_out=poses_out_data,
                             fps=60,
                             flight_logger=logger)
    try:
        rec.run(mav=m)
    finally:
        logger.close()
