import os
import threading
import time

import Prm.config as config

from payload_tracking.aruco_lib import MarkerPoseRecorder
from payload_tracking.color_track import ColorCircleRecorder


def start_camera(marker_size_m, video_out, csv_out, marker_ids=None,
                 capture_fps=48, frame_stride=1, camera_index=0,
                 ready_timeout=5, preview_port=None,
                 exposure_abs=5, gain=40,
                 width=2304, height=1536):
    os.makedirs(os.path.dirname(video_out) or ".", exist_ok=True)

    recorder = MarkerPoseRecorder(
        marker_size_m=marker_size_m,
        width=width,
        height=height,
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


def start_color_camera(circle_diameter_m, band_m, video_out, csv_out,
                       hue, hue_width, sat_min, val_min,
                       min_area_px=150, min_coverage_deg=200,
                       expected_range_m=None,
                       capture_fps=48, frame_stride=1, camera_index=0,
                       ready_timeout=5, preview_port=None,
                       exposure_abs=2, gain=1,
                       width=2304, height=1536):
    """
    Same as start_camera, tracking a colored ring instead of markers.

    The recorder it returns answers latest_detection() rather than
    latest_poses(), which is what estimator.latest_measurement dispatches on.
    """
    os.makedirs(os.path.dirname(video_out) or ".", exist_ok=True)

    recorder = ColorCircleRecorder(
        circle_diameter_m=circle_diameter_m,
        band_m=band_m,
        width=width,
        height=height,
        hue=hue,
        hue_width=hue_width,
        sat_min=sat_min,
        val_min=val_min,
        min_area_px=min_area_px,
        min_coverage_deg=min_coverage_deg,
        expected_range_m=expected_range_m,
        video_out=video_out,
        csv_out=csv_out,
        capture_fps=capture_fps,
        frame_stride=frame_stride,
        exposure_abs=exposure_abs,
        gain=gain,
        print_every=0,               # silent per-frame; close() still prints fps
        preview_port=preview_port)

    thread = threading.Thread(
        target=recorder.run,
        kwargs={"camera_index": camera_index},
        daemon=True)
    thread.start()

    # wait until the camera is actually streaming (frame_idx starts advancing)
    deadline = time.time() + ready_timeout
    while recorder.frame_idx == 0 and thread.is_alive() and time.time() < deadline:
        time.sleep(0.05)

    if recorder.frame_idx == 0:
        print("WARNING: camera did not start streaming, continuing without it")
    else:
        print(f"color recording started (pose t0 wall = {recorder._t0:.4f})")

    return recorder, thread


def start_payload_camera(video_out, csv_out, preview_port=None,
                         camera_index=0):
    """
    Whichever tracker config.EKF_SOURCE selects, wired from config.

    Lets a flight script start the camera without caring which one is running,
    since the estimator dispatches on the same setting.
    """
    if config.EKF_SOURCE == "color":
        return start_color_camera(
            circle_diameter_m=config.CIRCLE_DIAMETER,
            band_m=config.CIRCLE_BAND,
            video_out=video_out,
            csv_out=csv_out,
            hue=config.CIRCLE_HUE,
            hue_width=config.CIRCLE_HUE_WIDTH,
            sat_min=config.CIRCLE_SAT_MIN,
            val_min=config.CIRCLE_VAL_MIN,
            min_area_px=config.CIRCLE_MIN_AREA_PX,
            min_coverage_deg=config.CIRCLE_MIN_COVERAGE_DEG,
            expected_range_m=config.TETHER_LEN,
            capture_fps=config.CAM_FPS,
            frame_stride=config.CAM_STRIDE,
            camera_index=camera_index,
            preview_port=preview_port,
            gain=config.CAM_GAIN,
            exposure_abs=config.CAM_EXP_ABS,
            width=config.CAM_WIDTH,
            height=config.CAM_HEIGHT)

    marker_ids = [config.LEFT_MARKER_ID, config.CENTER_MARKER_ID,
                  config.RIGHT_MARKER_ID]

    return start_camera(
        marker_size_m=config.MARKER_EDGE_LEN,
        video_out=video_out,
        csv_out=csv_out,
        marker_ids=marker_ids,
        preview_port=preview_port,
        capture_fps=config.CAM_FPS,
        frame_stride=config.CAM_STRIDE,
        camera_index=camera_index,
        gain=config.CAM_GAIN,
        exposure_abs=config.CAM_EXP_ABS,
        width=config.CAM_WIDTH,
        height=config.CAM_HEIGHT)
