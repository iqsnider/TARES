# mission control imports
import comms.common as comms

# autonomy research imports
import comms.estimator as est
import Prm.config as config
import comms.camera as cam

# logging
from logs.flight import FlightLogger

# math
from datetime import datetime
import time


if __name__ == '__main__':
    connection = "udp:127.0.0.1:14550"
    control_freq = config.CONTROL_FREQUENCY

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/camera_fixing"
    video_out = f"{data_dir}/recording.avi"
    poses_out = f"{data_dir}/poses.csv"

    # mav connection
    m = comms.connect(connection)

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

