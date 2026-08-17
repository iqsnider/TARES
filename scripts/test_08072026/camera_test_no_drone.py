"""
Pure camera test: record ArUco detections and video, no flight data.

Runs until Ctrl+C. Use this to check detection rate at the operating distance
and to confirm a gain/exposure pair chosen by the tuning sweep.
"""
from datetime import datetime

from payload_tracking.aruco_lib import MarkerPoseRecorder
import Prm.config as config


if __name__ == "__main__":
    # live view config
    preview_port = 8080  # 8080 # live view; set None to disable
    track_marker_ids = [config.LEFT_MARKER_ID, config.CENTER_MARKER_ID,
                        config.RIGHT_MARKER_ID]   # only log these ids; None = all

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/test_08072026/camera_test_{stamp}"
    video_out = f"data/recording.avi"
    poses_out = f"data/poses.csv"

    rec = MarkerPoseRecorder(marker_size_m=config.MARKER_EDGE_LEN,
                             video_out=video_out,
                             csv_out=poses_out,
                             capture_fps=48,
                             frame_stride=1,
                             marker_ids=track_marker_ids,
                             preview_port=preview_port,
                             gain=config.CAM_GAIN,
                             exposure_abs=config.CAM_EXP_ABS)

    # run() owns the loop and installs its own SIGINT handler; close() flushes
    # the video and pose CSV on the way out
    rec.run()
