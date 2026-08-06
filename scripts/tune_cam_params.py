"""
Automated camera exposure/gain sweep for the payload ArUco tracker.

Records a short clip at each (exposure, gain) combination, scores it by ArUco
detection rate, and prints the winner. Nothing is kept: each clip is written to
a temp directory, scored, and deleted immediately, so peak disk use is one
8-second recording rather than the whole sweep.

Detection rate is the only metric that decides -- a dark noisy frame detects
fine, a bright smeared one does not.

RUN THIS AT THE OPERATING DISTANCE, IN THE OPERATING LIGHT, WITH THE CAMERA
MOVING. Settings tuned indoors at 2 m will not hold outdoors at 8 m: daylight
is orders of magnitude brighter and the marker shrinks from ~109 px to ~29 px.
A stationary camera has no motion blur, so every exposure scores the same and
the sweep tells you nothing.
"""
import csv
import os
import shutil
import tempfile
import time

import cv2
import numpy as np

import comms.camera as cam
import Prm.config as config


# units of 100us: 2 = 0.2 ms ... 40 = 4 ms
EXPOSURES = [10, 20, 40, 60, 80, 100]
GAINS = [20, 40, 70]

DWELL_S = 8.0          # recording time per combination
SETTLE_S = 2.0         # let the device settle between runs


def score_run(poses_path, video_path, fx, marker_size_m):
    """
    Detection rate and image stats for one recorded clip.

    poses.csv has one row per detected marker and one all-NaN row for frames
    with nothing, so frames are recovered by grouping on the frame column.
    """
    per_frame = {}
    ranges = []
    with open(poses_path) as f:
        for row in csv.DictReader(f):
            idx = int(row["frame"])
            hit = row["marker_id"] not in ("", "nan")
            per_frame[idx] = per_frame.get(idx, 0) + (1 if hit else 0)
            if hit:
                ranges.append(float(row["range_m"]))

    n_frames = len(per_frame)
    n_detected = sum(1 for v in per_frame.values() if v > 0)
    detect_rate = n_detected / n_frames if n_frames else 0.0
    markers_per_frame = sum(per_frame.values()) / n_frames if n_frames else 0.0
    mean_range = float(np.mean(ranges)) if ranges else float("nan")

    # apparent marker size drives everything: below ~3 px per module (18 px for
    # a 6x6 DICT_4X4 marker) detection collapses regardless of exposure
    marker_px = fx * marker_size_m / mean_range if ranges else float("nan")

    # sample the recording for exposure level
    luma = []
    capture = cv2.VideoCapture(video_path)
    i = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if i % 25 == 0:
            luma.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
        i += 1
    capture.release()

    return {"exposure_abs": None,
            "gain": None,
            "detect_rate": detect_rate,
            "markers_per_frame": markers_per_frame,
            "mean_range_m": mean_range,
            "marker_px": marker_px,
            "luma": float(np.mean(luma)) if luma else float("nan")}


if __name__ == "__main__":
    track_marker_ids = [config.LEFT_MARKER_ID,
                        config.CENTER_MARKER_ID,
                        config.RIGHT_MARKER_ID]

    # scratch space: cleared as we go and removed at the end
    scratch = tempfile.mkdtemp(prefix="camera_tuning_")

    combos = [(e, g) for e in EXPOSURES for g in GAINS]
    print(f"sweeping {len(combos)} combinations x {DWELL_S:.0f}s "
          f"= ~{len(combos)*(DWELL_S+5)/60:.1f} min")
    print("move the camera during the sweep so motion blur is exercised\n")

    results = []
    try:
        for n, (exposure, gain) in enumerate(combos, 1):
            video_out = os.path.join(scratch, "clip.avi")
            poses_out = os.path.join(scratch, "clip.csv")

            recorder, thread = cam.start_camera(
                marker_size_m=config.MARKER_EDGE_LEN,
                video_out=video_out,
                csv_out=poses_out,
                marker_ids=track_marker_ids,
                preview_port=None,   # preview costs a resize + encode per frame
                capture_fps=48,
                frame_stride=1,
                exposure_abs=exposure,
                gain=gain)

            time.sleep(DWELL_S)
            recorder.stop()
            # close() drains the encode queue before releasing the device. A
            # short timeout here leaves the previous recorder still holding
            # /dev/video0, and the next open() then fails during warm-up.
            thread.join(timeout=30)

            if recorder.frame_idx == 0 or thread.is_alive():
                print(f"[{n:2d}/{len(combos)}] exposure={exposure:3d} "
                      f"gain={gain:3d}  ->  camera did not start, skipped")
                time.sleep(SETTLE_S)
                continue

            row = score_run(poses_out, video_out, recorder.mtx[0, 0],
                            config.MARKER_EDGE_LEN)
            row["exposure_abs"] = exposure
            row["gain"] = gain
            results.append(row)

            # drop the clip before the next one: peak disk use stays at one
            os.remove(video_out)
            os.remove(poses_out)

            print(f"[{n:2d}/{len(combos)}] exposure={exposure:3d} "
                  f"({exposure/10:4.1f} ms) gain={gain:3d}  ->  "
                  f"detect {100*row['detect_rate']:5.1f}%  luma {row['luma']:5.1f}")

            time.sleep(SETTLE_S)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    if not results:
        raise SystemExit("no combination produced a recording")

    # best = highest detection rate, shortest exposure breaks ties (less blur)
    results.sort(key=lambda r: (-r["detect_rate"], r["exposure_abs"]))
    best = results[0]

    print()
    print("=" * 60)
    print(f"  exposure_abs = {best['exposure_abs']}   "
          f"({best['exposure_abs']/10:.1f} ms)")
    print(f"  gain         = {best['gain']}")
    print("=" * 60)
    print(f"{100*best['detect_rate']:.1f}% detection, "
          f"{best['markers_per_frame']:.2f} markers/frame, "
          f"luma {best['luma']:.0f}, "
          f"marker {best['marker_px']:.0f} px at {best['mean_range_m']:.1f} m")

    if best["luma"] > 200:
        print("\nWARNING: luma high -- likely clipping. Extend EXPOSURES "
              "shorter or GAINS lower and re-run.")
    if best["detect_rate"] > 0.98:
        print("\nNOTE: near-ceiling detection means this scene is too easy to "
              "discriminate settings. Re-run further from the markers.")
