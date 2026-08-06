import os
import threading
import time

from payload_tracking.aruco_lib import MarkerPoseRecorder


def start_camera(marker_size_m, video_out, csv_out, marker_ids=None,
                 capture_fps=48, frame_stride=1, camera_index=0,
                 ready_timeout=5, preview_port=None,
                 exposure_abs=5, gain=40):
    os.makedirs(os.path.dirname(video_out) or ".", exist_ok=True)

    recorder = MarkerPoseRecorder(
        marker_size_m=marker_size_m,
        video_out=video_out,
        csv_out=csv_out,
        capture_fps=capture_fps,     # MJPG at 2304x1536 offers 48 and nothing else
        frame_stride=frame_stride,   # 2 -> 24 fps, 3 -> 16, 4 -> 12
        marker_ids=marker_ids,       # track only these IDs (None = all)
        exposure_abs=exposure_abs,   # units of 100us; 5 = 0.5 ms
        gain=gain,
        flight_logger=None,          # no flight-logger integration (by design:
                                     # the control loop pumps the same mavlink
                                     # connection, and pymavlink is not
                                     # thread-safe)
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
        print(f"camera recording started (pose t0 wall = {recorder._t0:.4f})")

    return recorder, thread
