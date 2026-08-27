"""
Point the camera at the payload and read back what color it actually is.

Hue is a property of the tape, the light and the exposure together, not of the
tape alone: the same red reads near 0 in sun at a short exposure and near 20
indoors at high gain. So it has to be measured wherever the aircraft is about
to fly, at the exposure and gain it is about to fly with.

Prints the distribution of the saturated pixels it can see, and the config
lines that would match them.
"""

# autonomy research imports
import Prm.config as config
from payload_tracking.aruco_lib import apply_camera_controls

# math
import time
import cv2
import numpy as np


FRAMES = 20
SAT_FLOOR = 80        # below this a pixel is background, not tape


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.CAM_FPS)

    # measure at the settings the aircraft will fly with, or the numbers lie
    print(apply_camera_controls(0, config.CAM_EXP_ABS, config.CAM_GAIN))
    print(f"exposure {config.CAM_EXP_ABS} gain {config.CAM_GAIN}, "
          f"averaging {FRAMES} frames")

    hues, sats, vals = [], [], []
    for _ in range(FRAMES):
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        keep = s >= SAT_FLOOR
        hues.append(h[keep])
        sats.append(s[keep])
        vals.append(v[keep])
        time.sleep(0.05)
    cap.release()

    h = np.concatenate(hues)
    s = np.concatenate(sats)
    v = np.concatenate(vals)
    if len(h) < 100:
        print("almost nothing saturated in view, so there is nothing to fit")
        raise SystemExit

    # the biggest cluster of saturated hue is the target, since the background
    # outdoors is washed out and lands under the floor
    counts = np.bincount(h, minlength=180)
    peak = int(np.argmax(counts))
    near = np.minimum(np.abs(h.astype(int) - peak), 180 - np.abs(h.astype(int) - peak))
    band = near <= 20

    print(f"\n{len(h)} saturated pixels, peak hue {peak}")
    print(f"  hue  5/50/95 {np.percentile(h[band], [5, 50, 95]).round(0)}")
    print(f"  sat  5/50/95 {np.percentile(s[band], [5, 50, 95]).round(0)}")
    print(f"  val  5/50/95 {np.percentile(v[band], [5, 50, 95]).round(0)}")

    width = int(np.ceil(np.percentile(near[band], 90))) + 3
    sat_min = int(max(60, np.percentile(s[band], 5) - 15))
    val_min = int(max(30, np.percentile(v[band], 5) - 20))

    print("\nconfig lines that would match it:")
    print(f"  CIRCLE_HUE = {peak}")
    print(f"  CIRCLE_HUE_WIDTH = {width}")
    print(f"  CIRCLE_SAT_MIN = {sat_min}")
    print(f"  CIRCLE_VAL_MIN = {val_min}")
    print("\nkeep the payload roughly filling the view, and nothing else "
          "brightly colored behind it")
