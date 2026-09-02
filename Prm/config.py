import json
import os

import numpy as np

import sim.transformations as tf

PI = np.pi
GRAVITY = 9.81  # [m/s^2]

# which physical airframe is currently mounted. Everything that differs
# between rigs lives in Prm/airframes/<name>.json, so a new one is that file
# copied and edited. EKF_TEST is the bench rig: IF1200 numbers under its own
# name, so a desk test is identifiable in the logs and can be retuned without
# touching the aircraft
AIRFRAME = "IF1200"  # or "AURELIA_X4", "IF1200", "EKF_TEST"

_airframe_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "airframes")


def load_airframe(airframe):
    """
    The per-airframe constants, from Prm/airframes/<airframe>.json
    """
    with open(os.path.join(_airframe_dir, f"{airframe}.json")) as f:
        values = json.load(f)

    return values


def airframes():
    """
    Every airframe with a file, for a caller replaying an old session
    """
    names = sorted(f[:-5] for f in os.listdir(_airframe_dir)
                   if f.endswith(".json"))

    return names


_af = load_airframe(AIRFRAME)

##### drone parameters #####

MASS_DRONE = _af["MASS_DRONE"]  # [kg]
J = np.diag(_af["J_DIAG"])
# C_T = 8.7e-5
MAX_OMEGA = _af["MAX_OMEGA"]

# geometry
ARM_LEN = _af["ARM_LEN"]  # [m]
D = ARM_LEN / np.sqrt(2)
K_M = _af["K_M"]

###### payload parameters #####
MASS_TETHER = _af["MASS_TETHER"]  # [kg]
MASS_PAYLOAD = _af["MASS_PAYLOAD"]  # [kg]
DISC_DIAMETER = 0.3  # [m]
DISC_WIDTH = 0.0025  # [m]
TETHER_LEN = _af["TETHER_LEN"]  # [m]

# [m] tether pivot offset from drone CG in body frame
TETHER_PIVOT_OFFSET = np.zeros(3)


##### totals #####
MASS_PAYLOAD_EFF = MASS_PAYLOAD + MASS_TETHER
MASS_TOTAL = MASS_DRONE + MASS_PAYLOAD_EFF


##### camera tracking parameters #####

# settings
CAM_GAIN = 1  # 1
CAM_EXP_ABS = 3  # 3
CAM_PREVIEW_PORT = 8080  # None  # 8080
CAM_FPS = 48
CAM_STRIDE = 1
# must be a size the driver actually lists, since v4l2 snaps a bad one
# silently and the recorder then asserts on what it negotiated
CAM_WIDTH = 2304
CAM_HEIGHT = 1536


# camera offsets, referenced to the front of the drone and CoG [m]
CAM_OFFSET_X = _af["CAM_OFFSET_X"]
CAM_OFFSET_Y = _af["CAM_OFFSET_Y"]
CAM_OFFSET_Z = _af["CAM_OFFSET_Z"]

# camera rotation
# rad, about the nose axis (+Y, front)
CAM_ROLL = np.radians(_af["CAM_ROLL_DEG"])
# rad, about the lateral axis (+X, starboard)
CAM_PITCH = np.radians(_af["CAM_PITCH_DEG"])
# rad, about vertical axis (+Z, top), right hand rule (0 when up on the camera frame is in the drone nose direction)
CAM_YAW = np.radians(_af["CAM_YAW_DEG"])

# rotation to the body frame from the camera frame
CAM_R = tf.T_BC(CAM_ROLL, CAM_PITCH, CAM_YAW)


# payload marker parameters
MARKER_EDGE_LEN = 0.17  # [m]
MARKER_CENTER_TO_CENTER_DIST = 0.26  # [m]
LEFT_MARKER_ID = 30
CENTER_MARKER_ID = 245
RIGHT_MARKER_ID = 233

# payload color ring parameters (payload_tracking/color_track.py)
CIRCLE_DIAMETER = 0.31  # [m] outer diameter of the ring
CIRCLE_BAND = 0.015  # [m] width of the colored band
# hue is 0-179 in OpenCV, and red straddles 0, which color_mask wraps for.
# Dry grass and bare soil sit close to red in hue, so saturation is what
# separates the tape from the ground rather than hue
CIRCLE_HUE = 0  # red
CIRCLE_HUE_WIDTH = 12
CIRCLE_SAT_MIN = 130
CIRCLE_VAL_MIN = 50
CIRCLE_MIN_AREA_PX = 150
# how far around the circle the visible color must wrap, in degrees, measured
# about the fitted center. The tether and the payload below cut pieces out of
# the ring, and a circle through too short an arc is badly conditioned: on the
# 0828 bench run the fitted radius holds to +-12 px above 180 deg and spreads
# to +-29 px below it. Measured about the centroid this number reads high,
# since a centroid on the arc has points on every side of it
CIRCLE_MIN_COVERAGE_DEG = 70


##### mission parameters ######
CRUISE_SPEED = 1
GRID_X_START = 0
GRID_Y_START = 0
GRID_X_END = 10
GRID_Y_END = 10
GRID_ALTITUDE = 15

##### control parameters #####
CONTROL_FREQUENCY = 50

# outer-loop LQR cost weights, by controller (sim/dynamics.py). The position
# weights are stored as the tolerance they came from, so a _TOL of 1.5 m is a
# weight of 1/1.5^2

# drone-only outer loop (OuterLoopLQR)
LQR_DRONE_Q_POS_XY = _af["LQR_DRONE_Q_POS_XY"]
LQR_DRONE_Q_POS_Z = _af["LQR_DRONE_Q_POS_Z"]
LQR_DRONE_Q_VEL_XY = _af["LQR_DRONE_Q_VEL_XY"]
LQR_DRONE_Q_VEL_Z = _af["LQR_DRONE_Q_VEL_Z"]
LQR_DRONE_R_ACC = _af["LQR_DRONE_R_ACC"]

# drone outer loop with integral action (OuterLoopLQI)
LQI_DRONE_Q_POS_XY = _af["LQI_DRONE_Q_POS_XY"]
LQI_DRONE_Q_POS_Z = _af["LQI_DRONE_Q_POS_Z"]
LQI_DRONE_Q_VEL_XY = _af["LQI_DRONE_Q_VEL_XY"]
LQI_DRONE_Q_VEL_Z = _af["LQI_DRONE_Q_VEL_Z"]
LQI_DRONE_Q_INT_XY = (1/_af["LQI_DRONE_Q_INT_XY_TOL"])**2
LQI_DRONE_Q_INT_Z = (1/_af["LQI_DRONE_Q_INT_Z_TOL"])**2
LQI_DRONE_R_ACC = _af["LQI_DRONE_R_ACC"]
# error norm [m] below which the integrator is allowed to accumulate
LQI_DRONE_E_BAND = _af["LQI_DRONE_E_BAND"]
# cap on the integral term's own contribution to u [m/s^2]
LQI_DRONE_U_I_MAX = _af["LQI_DRONE_U_I_MAX"]

# payload outer loop (OuterLoopPayloadLQR)
LQR_PAYLOAD_W_POS_XY = (1/_af["LQR_PAYLOAD_W_POS_XY_TOL"])**2
LQR_PAYLOAD_W_POS_Z = (1/_af["LQR_PAYLOAD_W_POS_Z_TOL"])**2
LQR_PAYLOAD_TUNING_CONST = _af["LQR_PAYLOAD_TUNING_CONST"]
# the lag state changes what the input cost buys, so the lag aware design
# carries its own tuning constant rather than sharing the one above
LQR_PAYLOAD_LAG_TUNING_CONST = _af["LQR_PAYLOAD_LAG_TUNING_CONST"]

# payload outer loop with integral action (OuterLoopPayloadLQI)
LQI_PAYLOAD_W_POS_XY = (1/_af["LQI_PAYLOAD_W_POS_XY_TOL"])**2
LQI_PAYLOAD_W_POS_Z = (1/_af["LQI_PAYLOAD_W_POS_Z_TOL"])**2
LQI_PAYLOAD_W_INT_XY = (1/_af["LQI_PAYLOAD_W_INT_XY_TOL"])**2
LQI_PAYLOAD_W_INT_Z = (1/_af["LQI_PAYLOAD_W_INT_Z_TOL"])**2
LQI_PAYLOAD_TUNING_CONST = _af["LQI_PAYLOAD_TUNING_CONST"]
LQI_PAYLOAD_LAG_TUNING_CONST = _af["LQI_PAYLOAD_LAG_TUNING_CONST"]

# error norm [m] below which the integrator is allowed to accumulate
LQI_PAYLOAD_E_BAND = _af["LQI_PAYLOAD_E_BAND"]
# cap on the integral term's own contribution to u [m/s^2], so it cannot
# run away even while accumulating. Sized per-airframe from each rig's own
# commanded-acceleration logs (p95 of a clean closed-loop flight).
LQI_PAYLOAD_U_I_MAX = _af["LQI_PAYLOAD_U_I_MAX"]

##### airframe acceleration response (the WithLag controllers) #####

# how long the airframe takes to deliver a commanded acceleration, a first
# order lag fit to commanded against measured accel in flight [s]
ACCEL_LAG_TAU = _af["ACCEL_LAG_TAU"]

##### payload swing EKF tuning (sim/estimation/ekf.py) #####

# which tracker feeds the filter, "aruco" or "color". The color ring gives no
# payload yaw, so psi_p goes unobserved and only the swing angles are measured
EKF_SOURCE = "color"


def ekf_tuning_for(airframe):
    """
    Filter tuning for a named airframe, for replaying a session recorded
    before the tuning was snapshotted with it.

    The file stores the noise as standard deviations and the angles in
    degrees, which is what the filter is tuned in; the EKF wants variances
    and radians.
    """
    af = load_airframe(airframe)
    tuning = {"q_xy": af["EKF_Q_XY_STD"]**2,
              "q_yaw": af["EKF_Q_YAW_STD"]**2,
              "zeta": af["EKF_ZETA"],
              "sigma_xy": np.radians(af["EKF_SIGMA_XY_DEG"]),
              "sigma_yaw": np.radians(af["EKF_SIGMA_YAW_DEG"]),
              "sigma_alpha_0": np.radians(af["EKF_SIGMA_ALPHA_0_DEG"]),
              "sigma_rate_0": np.radians(af["EKF_SIGMA_RATE_0_DEG"]),
              "sigma_psi_p_0": np.radians(af["EKF_SIGMA_PSI_P_0_DEG"])}

    return tuning


_ekf = ekf_tuning_for(AIRFRAME)

# process noise on alpha_ddot_xy and on psi_p
EKF_Q_XY = _ekf["q_xy"]
EKF_Q_YAW = _ekf["q_yaw"]

# swing damping ratio. 0 is the undamped pendulum the model started as; 0.2
# measured out better than 0 at every prediction horizon on both 0819 payload
# flights. Get it per rig from the free decay of a released swing
EKF_ZETA = _ekf["zeta"]

# measurement noise: bearing and payload yaw [rad]
EKF_SIGMA_XY = _ekf["sigma_xy"]
EKF_SIGMA_YAW = _ekf["sigma_yaw"]

# initial 1-sigma on swing angle [rad], swing rate [rad/s], payload yaw [rad]
EKF_SIGMA_ALPHA_0 = _ekf["sigma_alpha_0"]
EKF_SIGMA_RATE_0 = _ekf["sigma_rate_0"]
EKF_SIGMA_PSI_P_0 = _ekf["sigma_psi_p_0"]


##### transmitter control tuning #####
STICK_TRIM = _af["STICK_TRIM"]
STICK_TRAVEL = _af["STICK_TRAVEL"]
STICK_DZ = _af["STICK_DZ"]
MAX_STICK_ACCELERATION = _af["MAX_STICK_ACCELERATION"]
PAYLOAD_V_XY_MAX = _af["PAYLOAD_V_XY_MAX"]  # m/s
PAYLOAD_V_Z_MAX = _af["PAYLOAD_V_Z_MAX"]  # m/s
