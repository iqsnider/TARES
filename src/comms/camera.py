import os
import threading
import time

from payload_tracking.aruco_lib import MarkerPoseRecorder


def start_camera(marker_size_m, video_out, csv_out, marker_ids=None, fps=48, camera_index=0, ready_timeout=5, preview_port=None):
    os.makedirs(os.path.dirname(video_out) or ".", exist_ok=True)

    recorder = MarkerPoseRecorder(
        marker_size_m=marker_size_m,
        video_out=video_out,
        csv_out=csv_out,
        fps=fps,
        marker_ids=marker_ids,       # track only these IDs (None = all)
        flight_logger=None,          # no flight-logger integration (by design)
        print_every=0,               # silent per-frame; close() still prints fps
        preview_port=preview_port)   # None = no live view

    thread = threading.Thread(
        target=recorder.run,
        kwargs={"camera_index": camera_index, "mav": None},
        daemon=True)
    thread.start()

    # wait until the camera is actually streaming (frame_idx starts advancing)
    deadline = time.time() + ready_timeout
    while recorder.frame_idx == 0 and thread.is_alive() and time.time() < deadline:
        time.sleep(0.05)

    if recorder.frame_idx == 0:
        print("WARNING: camera did not start streaming -- continuing without it")
    else:
        # recorder._t0 is the wall-clock instant (time.time()) that the pose
        # CSV's time_s column is measured from. Printing it here gives you the
        # absolute anchor to line the pose CSV up against the flight log's
        # wall_time column later, without merging the two files yet.
        print(f"camera recording started (pose t0 wall = {recorder._t0:.4f})")
    return recorder, thread
