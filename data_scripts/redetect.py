"""
Replay a flight's recorded video through the ring detector.

The camera and the flight are done, but the frames are not: the detector's
thresholds and its range gate can be retried on the ground as many times as it
takes. Writes a new circles CSV next to the original so run_on_log.py can be
pointed at either one.

Set the run and the overrides below and run it.
"""

import csv
import glob
import json
import os

import cv2
import numpy as np

from payload_tracking.color_track import ColorCircleRecorder

NAN = float("nan")


def build_detector(snap, **overrides):
    """
    A recorder built only to detect: the camera is never opened, so it takes
    the geometry from the session's own snapshot and nothing else.
    """
    params = dict(circle_diameter=snap["CIRCLE_DIAMETER"],
                  band=snap["CIRCLE_BAND"],
                  hue=snap["CIRCLE_HUE"],
                  hue_width=snap["CIRCLE_HUE_WIDTH"],
                  sat_min=snap["CIRCLE_SAT_MIN"],
                  val_min=snap["CIRCLE_VAL_MIN"],
                  min_area_px=snap["CIRCLE_MIN_AREA_PX"],
                  min_coverage_deg=snap["CIRCLE_MIN_COVERAGE_DEG"],
                  expected_range=snap["TETHER_LEN"])
    params.update(overrides)

    return ColorCircleRecorder(**params)


def frame_times(session):
    """
    The wall clock of each frame, off the original CSV, so a replayed run keeps
    the timing the flight log is aligned to
    """
    path = os.path.join(session, "circles.csv")
    if not os.path.exists(path):
        return {}

    times = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            times[int(row["frame"])] = (row["wall_time"], row["time_s"])

    return times


def redetect(session, csv_name="circles_redetect.csv", **overrides):
    """
    Run every frame of the session's video through the detector and write the
    result. Returns the detection rate before and after.
    """
    video = os.path.join(session, "recording.avi")
    snap = json.load(open(os.path.join(session, "config_snapshot.json")))
    det = build_detector(snap, **overrides)
    times = frame_times(session)

    cap = cv2.VideoCapture(video)
    out_path = os.path.join(session, csv_name)
    hits = 0
    n = 0

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "wall_time", "time_s", "u_px", "v_px",
                    "radius_px", "coverage_deg", "n_px", "x", "y", "z",
                    "range_m"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            d = det.detect(frame)
            wall, t = times.get(n, (f"{n/snap['CAM_FPS']:.4f}",
                                    f"{n/snap['CAM_FPS']:.4f}"))
            if d is None:
                w.writerow([n, wall, t] + [NAN]*9)
            else:
                x, y, z = d["p_C"]
                w.writerow([n, wall, t, f"{d['u']:.2f}", f"{d['v']:.2f}",
                            f"{d['radius']:.2f}", f"{d['coverage']:.1f}",
                            d["n_px"], f"{x:.6f}", f"{y:.6f}", f"{z:.6f}",
                            f"{d['range_m']:.6f}"])
                hits += 1
            n += 1

    cap.release()

    was = np.nan
    if times:
        import pandas as pd
        was = pd.read_csv(os.path.join(session, "circles.csv"))["radius_px"].notna().mean()

    return 100*was, 100*hits/max(n, 1), n


if __name__ == '__main__':
    runs = sorted(glob.glob("data/test_09022026/*icra_test*/"))

    # what to change from what the run actually flew. The ring's own diameter
    # is the one to reach for first: it scales every computed range, and the
    # detector throws away any blob whose range misses the tether by more than
    # range_tol, so a diameter that is too big pushes good frames out of the
    # gate
    overrides = dict(circle_diameter=0.35,
                     range_tol=0.5)

    print(f"{'run':<40}{'was':>7}{'now':>7}{'frames':>9}")
    for session in runs:
        if not os.path.exists(os.path.join(session, "recording.avi")):
            print(f"{os.path.basename(session.rstrip('/')):<40}  no video")
            continue
        was, now, n = redetect(session, **overrides)
        print(f"{os.path.basename(session.rstrip('/')):<40}{was:>6.1f}%"
              f"{now:>6.1f}%{n:>9}")
