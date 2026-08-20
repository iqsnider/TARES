"""
Mainly just for seeing what RC values come up in the data
"""

# mission control imports
import comms.common as comms
from comms.payload_autopilot import StickControl

# autonomy research imports
import sim.dynamics as dynamics
import Prm.config as config
import comms.camera as cam

# logging
from logs.flight import FlightLogger


from datetime import datetime
import time


if __name__ == '__main__':
    connection = "/dev/ttyACM0"
    baud = 115200
    # connection = "udp:127.0.0.1:14550"
    # takeoff_altitude = 15
    control_freq = config.CONTROL_FREQUENCY

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/test_08192026/log_test_{stamp}"
    video_out = f"{data_dir}/recording.avi"
    poses_out = f"{data_dir}/poses.csv"

    # add baud here if connected to real drone
    m = comms.connect(connection, baud)

    # check for armability
    # comms.wait_until_armable(m)

    track_marker_ids = [config.LEFT_MARKER_ID, config.CENTER_MARKER_ID, config.RIGHT_MARKER_ID]   # only these are logged; others dropped
    # start camera first
    recorder, cam_thread = cam.start_camera(
        marker_size_m=config.MARKER_EDGE_LEN,
        video_out=video_out,
        csv_out=poses_out,
        marker_ids=track_marker_ids,
        preview_port=config.CAM_PREVIEW_PORT,
        capture_fps=config.CAM_FPS,
        frame_stride=config.CAM_STRIDE,
        gain=config.CAM_GAIN,
        exposure_abs=config.CAM_EXP_ABS)

    # intialize the logs
    logger = FlightLogger(data_dir=data_dir)
    try:
        while True:
            logger.pump(m)
            time.sleep(1/control_freq)

    finally:
        recorder.stop()
        cam_thread.join(timeout=5)
        logger.close()

