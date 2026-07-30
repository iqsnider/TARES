"""Which flight the analysis scripts point at.

One place to name the run, so the drone plots, the payload plots and the EKF
plots are all looking at the same flight. Change RUN/FLIGHT here and every
script follows.
"""
from pathlib import Path

DATA_ROOT = Path("~/TARES_SITL/data").expanduser()

# --- 23 Jul 2026, last test of the day ---------------------------------------
TEST_DIR = DATA_ROOT / "test_07232026" / "all_data_07232026"
RUN = "step_test_with_camera_20260723_114555"
POSE = TEST_DIR / RUN / "poses.csv"
FLIGHT = TEST_DIR / "flight_20260723_114556.csv"

# pose clock -> flight clock [s]. Found by minimising NIS over (offset, L) in
# run_on_log.py; keep every script on the same value or the payload positions
# they draw will not agree.
POSE_TIME_OFFSET = 0.17

# tether length for the dynamics / from the pivot to the marker board [m]
L_DYN, L_MARKER = 8.0, 8.31

CALIB = Path("~/TARES_SITL/src/payload_tracking/"
             "camera_calibration/calibration.json").expanduser()

# written alongside the run the figures came from
OUT_DIR = TEST_DIR / RUN / "ekf"


def check():
    """Fail early, with the missing path named, rather than deep in pandas."""
    for p in (POSE, FLIGHT, CALIB):
        if not p.exists():
            raise FileNotFoundError(p)
