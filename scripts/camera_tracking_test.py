from pymavlink import mavutil

from payload_tracking.aruco_lib import MarkerPoseRecorder

import comms.common as comms
import comms.control as control

from logs.flight import FlightLogger

if __name__ == "__main__":
    # connection = "/dev/ttyACM0"
    # baud = 115200
    connection = "udp:127.0.0.1:14550"
    control_freq = 50

    # add baud here if connected to real drone
    m = comms.connect(connection)
    control.request_fast_state(m, hz=control_freq)

    # initialize drone logger
    logger = FlightLogger(data_dir="camera_test_data")

    # begin payload recording
    rec = MarkerPoseRecorder(marker_size_m=0.14,
                             video_out="recording_140mm.avi",
                             csv_out="poses_140mm.csv",
                             fps=60)
    try:
        rec.run(mav=m)
    finally:
        logger.close()
