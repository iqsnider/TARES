import numpy as np

import sim.transformations as tf

PI = np.pi
GRAVITY = 9.81  # [m/s^2]

# which physical airframe is currently mounted. Selects every per-airframe
# EKF_TEST is the bench rig: IF1200 numbers under its own name, so a desk test
# is identifiable in the logs and can be retuned without touching the aircraft
AIRFRAME = "EKF_TEST"  # or "AURELIA_X4", "EKF_TEST"

##### drone parameters #####

_mass_drone_by_airframe = {"AURELIA_X4": 2.55, "IF1200": 17,
                           "EKF_TEST": 17}  # [kg]
MASS_DRONE = _mass_drone_by_airframe[AIRFRAME]

_j_by_airframe = {"AURELIA_X4": np.diag([0.080, 0.080, 0.129]),
                  "IF1200": np.diag([0.080, 0.080, 0.129]),
                  "EKF_TEST": np.diag([0.080, 0.080, 0.129])}
J = _j_by_airframe[AIRFRAME]
# C_T = 8.7e-5
_max_omega_by_airframe = {"AURELIA_X4": 1100, "IF1200": 1100,
                          "EKF_TEST": 1100}
MAX_OMEGA = _max_omega_by_airframe[AIRFRAME]

# geometry
_arm_len_by_airframe = {"AURELIA_X4": 0.381, "IF1200": 0.381,
                        "EKF_TEST": 0.381}  # [m]
ARM_LEN = _arm_len_by_airframe[AIRFRAME]
D = ARM_LEN / np.sqrt(2)
_k_m_by_airframe = {"AURELIA_X4": 0.02, "IF1200": 0.02, "EKF_TEST": 0.02}
K_M = _k_m_by_airframe[AIRFRAME]

###### payload parameters #####
_mass_tether_by_airframe = {"AURELIA_X4": 0.68,
                            "IF1200": 0.9,
                            "EKF_TEST": 0.9}  # [kg]
MASS_TETHER = _mass_tether_by_airframe[AIRFRAME]  # [kg]
_mass_payload_by_airframe = {"AURELIA_X4": 1.13, "IF1200": 6.0,
                             "EKF_TEST": 6.0}  # [kg]
MASS_PAYLOAD = _mass_payload_by_airframe[AIRFRAME]
DISC_DIAMETER = 0.3  # [m]
DISC_WIDTH = 0.0025  # [m]
_tether_len_by_airframe = {"AURELIA_X4": 10.2, "IF1200": 8.1,
                           "EKF_TEST": 5.5}  # [m]
TETHER_LEN = _tether_len_by_airframe[AIRFRAME]

# [m] tether pivot offset from drone CG in body frame
TETHER_PIVOT_OFFSET = np.zeros(3)


##### totals #####
MASS_PAYLOAD_EFF = MASS_PAYLOAD + MASS_TETHER
MASS_TOTAL = MASS_DRONE + MASS_PAYLOAD_EFF


##### camera tracking parameters #####

# settings
CAM_GAIN = 10
_cam_exp_abs_by_airframe = {"AURELIA_X4": 2, "IF1200": 2, "EKF_TEST": 150}
CAM_EXP_ABS = _cam_exp_abs_by_airframe[AIRFRAME]
CAM_PREVIEW_PORT = 8080  # None  # 8080
CAM_FPS = 48
CAM_STRIDE = 1
# must be a size the driver actually lists, since v4l2 snaps a bad one
# silently and the recorder then asserts on what it negotiated
CAM_WIDTH = 2304
CAM_HEIGHT = 1536


# camera offsets, referenced to the front of the drone and CoG [m]
_cam_offset_x_by_airframe = {"AURELIA_X4": -0.34, "IF1200": -0.1,
                             "EKF_TEST": 0}  # 0.11
_cam_offset_y_by_airframe = {"AURELIA_X4": 0.12, "IF1200": -0.154,
                             "EKF_TEST": 0}  # 0.22
_cam_offset_z_by_airframe = {"AURELIA_X4": -0.05, "IF1200": -0.1,
                             "EKF_TEST": 0}
CAM_OFFSET_X = _cam_offset_x_by_airframe[AIRFRAME]
CAM_OFFSET_Y = _cam_offset_y_by_airframe[AIRFRAME]
CAM_OFFSET_Z = _cam_offset_z_by_airframe[AIRFRAME]

# camera rotation

_cam_roll_by_airframe = {"AURELIA_X4": 0, "IF1200": 0, "EKF_TEST": 0}
_cam_pitch_by_airframe = {"AURELIA_X4": 0, "IF1200": 0, "EKF_TEST": 0}
_cam_yaw_by_airframe = {"AURELIA_X4": PI/2, "IF1200": PI, "EKF_TEST": 0}
# rad, about the nose axis (+Y, front)
CAM_ROLL = _cam_roll_by_airframe[AIRFRAME]
# rad, about the lateral axis (+X, starboard)
CAM_PITCH = _cam_pitch_by_airframe[AIRFRAME]
# rad, about vertical axis (+Z, top), right hand rule (0 when up on the camera frame is in the drone nose direction)
CAM_YAW = _cam_yaw_by_airframe[AIRFRAME]

# rotation to the body frame from the camera frame
CAM_R = tf.T_BC(CAM_ROLL, CAM_PITCH, CAM_YAW)


# payload marker parameters
MARKER_EDGE_LEN = 0.17  # [m]
MARKER_CENTER_TO_CENTER_DIST = 0.39  # [m]
LEFT_MARKER_ID = 30
CENTER_MARKER_ID = 245
RIGHT_MARKER_ID = 233

# payload color ring parameters (payload_tracking/color_track.py)
CIRCLE_DIAMETER = 0.31  # [m] outer diameter of the ring
CIRCLE_BAND = 0.11  # [m] width of the colored band
# hue is 0-179 in OpenCV, and red straddles 0, which color_mask wraps for.
# Dry grass and bare soil sit close to red in hue, so saturation is what
# separates the tape from the ground rather than hue
CIRCLE_HUE = 0  # red
CIRCLE_HUE_WIDTH = 12
CIRCLE_SAT_MIN = 130
CIRCLE_VAL_MIN = 50
CIRCLE_MIN_AREA_PX = 150
# how far around the circle the visible arc must wrap, in degrees. The tether
# and the payload below cut pieces out of the ring, and only the arc that is
# left gets fitted; below about 90 deg the center becomes badly conditioned
CIRCLE_MIN_COVERAGE_DEG = 200


##### mission parameters ######
CRUISE_SPEED = 1
GRID_X_START = 0
GRID_Y_START = 0
GRID_X_END = 10
GRID_Y_END = 10
GRID_ALTITUDE = 15

##### control parameters #####
CONTROL_FREQUENCY = 50

# outer-loop LQR cost weights, by controller (sim/dynamics.py)

# drone-only outer loop (OuterLoopLQR)
_lqr_drone_q_pos_xy_by_airframe = {"AURELIA_X4": 1, "IF1200": 1,
                                   "EKF_TEST": 1}
_lqr_drone_q_pos_z_by_airframe = {"AURELIA_X4": 1, "IF1200": 1,
                                  "EKF_TEST": 1}
_lqr_drone_q_vel_xy_by_airframe = {"AURELIA_X4": 1, "IF1200": 1,
                                   "EKF_TEST": 1}
_lqr_drone_q_vel_z_by_airframe = {"AURELIA_X4": 1, "IF1200": 1,
                                  "EKF_TEST": 1}
_lqr_drone_r_acc_by_airframe = {"AURELIA_X4": 1, "IF1200": 1, "EKF_TEST": 1}
LQR_DRONE_Q_POS_XY = _lqr_drone_q_pos_xy_by_airframe[AIRFRAME]
LQR_DRONE_Q_POS_Z = _lqr_drone_q_pos_z_by_airframe[AIRFRAME]
LQR_DRONE_Q_VEL_XY = _lqr_drone_q_vel_xy_by_airframe[AIRFRAME]
LQR_DRONE_Q_VEL_Z = _lqr_drone_q_vel_z_by_airframe[AIRFRAME]
LQR_DRONE_R_ACC = _lqr_drone_r_acc_by_airframe[AIRFRAME]

# drone outer loop with integral action (OuterLoopLQI)
_lqi_drone_q_pos_xy_by_airframe = {"AURELIA_X4": 1, "IF1200": 1, "EKF_TEST": 1}
_lqi_drone_q_pos_z_by_airframe = {"AURELIA_X4": 1, "IF1200": 1, "EKF_TEST": 1}
_lqi_drone_q_vel_xy_by_airframe = {"AURELIA_X4": 1, "IF1200": 1, "EKF_TEST": 1}
_lqi_drone_q_vel_z_by_airframe = {"AURELIA_X4": 1, "IF1200": 1, "EKF_TEST": 1}
_lqi_drone_q_int_xy_by_airframe = {"AURELIA_X4": (1/8)**2, "IF1200": (1/8)**2,
                                   "EKF_TEST": (1/8)**2}
_lqi_drone_q_int_z_by_airframe = {"AURELIA_X4": (1/8)**2, "IF1200": (1/8)**2,
                                  "EKF_TEST": (1/8)**2}
_lqi_drone_r_acc_by_airframe = {"AURELIA_X4": 1, "IF1200": 1, "EKF_TEST": 1}
LQI_DRONE_Q_POS_XY = _lqi_drone_q_pos_xy_by_airframe[AIRFRAME]
LQI_DRONE_Q_POS_Z = _lqi_drone_q_pos_z_by_airframe[AIRFRAME]
LQI_DRONE_Q_VEL_XY = _lqi_drone_q_vel_xy_by_airframe[AIRFRAME]
LQI_DRONE_Q_VEL_Z = _lqi_drone_q_vel_z_by_airframe[AIRFRAME]
LQI_DRONE_Q_INT_XY = _lqi_drone_q_int_xy_by_airframe[AIRFRAME]
LQI_DRONE_Q_INT_Z = _lqi_drone_q_int_z_by_airframe[AIRFRAME]
LQI_DRONE_R_ACC = _lqi_drone_r_acc_by_airframe[AIRFRAME]
# error norm [m] below which the integrator is allowed to accumulate
_lqi_drone_e_band_by_airframe = {"AURELIA_X4": 5, "IF1200": 5, "EKF_TEST": 5}
LQI_DRONE_E_BAND = _lqi_drone_e_band_by_airframe[AIRFRAME]
# cap on the integral term's own contribution to u [m/s^2]
_lqi_drone_u_i_max_by_airframe = {"AURELIA_X4": 1.5, "IF1200": 2, "EKF_TEST": 2}
LQI_DRONE_U_I_MAX = _lqi_drone_u_i_max_by_airframe[AIRFRAME]

# payload outer loop (OuterLoopPayloadLQR)
_lqr_payload_w_pos_xy_by_airframe = {
    "AURELIA_X4": (1/1)**2, "IF1200": (1/1.2)**2, "EKF_TEST": (1/1.2)**2}
_lqr_payload_w_pos_z_by_airframe = {
    "AURELIA_X4": (1/1)**2, "IF1200": (1/1.2)**2, "EKF_TEST": (1/1.2)**2}
_lqr_payload_tuning_const_by_airframe = {
    "AURELIA_X4": (1/1)**2, "IF1200": (1/1)**2, "EKF_TEST": (1/1)**2}
LQR_PAYLOAD_W_POS_XY = _lqr_payload_w_pos_xy_by_airframe[AIRFRAME]
LQR_PAYLOAD_W_POS_Z = _lqr_payload_w_pos_z_by_airframe[AIRFRAME]
LQR_PAYLOAD_TUNING_CONST = _lqr_payload_tuning_const_by_airframe[AIRFRAME]

# payload outer loop with integral action (OuterLoopPayloadLQI)
_lqi_payload_w_pos_xy_by_airframe = {
    "AURELIA_X4": (1/0.1)**2, "IF1200": (1/1.5)**2, "EKF_TEST": (1/1.5)**2}
_lqi_payload_w_pos_z_by_airframe = {
    "AURELIA_X4": (1/1)**2, "IF1200": (1/1.5)**2, "EKF_TEST": (1/1.5)**2}
_lqi_payload_w_int_xy_by_airframe = {
    "AURELIA_X4": (1/1)**2, "IF1200": (1/8)**2, "EKF_TEST": (1/8)**2}
_lqi_payload_w_int_z_by_airframe = {
    "AURELIA_X4": (1/4)**2, "IF1200": (1/8)**2, "EKF_TEST": (1/8)**2}
_lqi_payload_tuning_const_by_airframe = {
    "AURELIA_X4": (1/1)**2, "IF1200": (1/1)**2, "EKF_TEST": (1/1)**2}
LQI_PAYLOAD_W_POS_XY = _lqi_payload_w_pos_xy_by_airframe[AIRFRAME]
LQI_PAYLOAD_W_POS_Z = _lqi_payload_w_pos_z_by_airframe[AIRFRAME]
LQI_PAYLOAD_W_INT_XY = _lqi_payload_w_int_xy_by_airframe[AIRFRAME]
LQI_PAYLOAD_W_INT_Z = _lqi_payload_w_int_z_by_airframe[AIRFRAME]
LQI_PAYLOAD_TUNING_CONST = _lqi_payload_tuning_const_by_airframe[AIRFRAME]
# error norm [m] below which the integrator is allowed to accumulate
_lqi_payload_e_band_by_airframe = {"AURELIA_X4": 5, "IF1200": 5,
                                   "EKF_TEST": 5}
LQI_PAYLOAD_E_BAND = _lqi_payload_e_band_by_airframe[AIRFRAME]
# cap on the integral term's own contribution to u [m/s^2], so it cannot
# run away even while accumulating. Sized per-airframe from each rig's own
# commanded-acceleration logs (p95 of a clean closed-loop flight).
_lqi_payload_u_i_max_by_airframe = {"AURELIA_X4": 1.5, "IF1200": 2,
                                    "EKF_TEST": 2}
LQI_PAYLOAD_U_I_MAX = _lqi_payload_u_i_max_by_airframe[AIRFRAME]

##### payload swing EKF tuning (sim/estimation/ekf.py) #####

# which tracker feeds the filter, "aruco" or "color". The color ring gives no
# payload yaw, so psi_p goes unobserved and only the swing angles are measured
EKF_SOURCE = "color"

# process noise on alpha_ddot_xy and on psi_p
_ekf_q_xy_by_airframe = {"AURELIA_X4": (0.02)**2, "IF1200": (0.02)**2,
                         "EKF_TEST": (0.2)**2}
_ekf_q_yaw_by_airframe = {"AURELIA_X4": (0.3)**2, "IF1200": (0.3)**2,
                          "EKF_TEST": (0.3)**2}
EKF_Q_XY = _ekf_q_xy_by_airframe[AIRFRAME]
EKF_Q_YAW = _ekf_q_yaw_by_airframe[AIRFRAME]

# swing damping ratio. 0 is the undamped pendulum the model started as; 0.2
# measured out better than 0 at every prediction horizon on both 0819 payload
# flights. Get it per rig from the free decay of a released swing
_ekf_zeta_by_airframe = {"AURELIA_X4": 0, "IF1200": 0.05, "EKF_TEST": 0}
EKF_ZETA = _ekf_zeta_by_airframe[AIRFRAME]

# measurement noise: bearing and payload yaw [rad]
_ekf_sigma_xy_by_airframe = {"AURELIA_X4": np.radians(0.5),
                             "IF1200": np.radians(0.5),
                             "EKF_TEST": np.radians(0.5)}
_ekf_sigma_yaw_by_airframe = {"AURELIA_X4": np.radians(30),
                              "IF1200": np.radians(30),
                              "EKF_TEST": np.radians(30)}
EKF_SIGMA_XY = _ekf_sigma_xy_by_airframe[AIRFRAME]
EKF_SIGMA_YAW = _ekf_sigma_yaw_by_airframe[AIRFRAME]

# initial 1-sigma on swing angle [rad], swing rate [rad/s], payload yaw [rad]
_ekf_sigma_alpha_0_by_airframe = {"AURELIA_X4": np.radians(30),
                                  "IF1200": np.radians(30),
                                  "EKF_TEST": np.radians(30)}
_ekf_sigma_rate_0_by_airframe = {"AURELIA_X4": np.radians(30),
                                 "IF1200": np.radians(30),
                                 "EKF_TEST": np.radians(30)}
_ekf_sigma_psi_p_0_by_airframe = {"AURELIA_X4": np.radians(15),
                                  "IF1200": np.radians(15),
                                  "EKF_TEST": np.radians(15)}
EKF_SIGMA_ALPHA_0 = _ekf_sigma_alpha_0_by_airframe[AIRFRAME]
EKF_SIGMA_RATE_0 = _ekf_sigma_rate_0_by_airframe[AIRFRAME]
EKF_SIGMA_PSI_P_0 = _ekf_sigma_psi_p_0_by_airframe[AIRFRAME]


def ekf_tuning_for(airframe):
    """
    Filter tuning for a named airframe, for replaying a session recorded
    before the tuning was snapshotted with it.
    """
    tuning = {"q_xy": _ekf_q_xy_by_airframe[airframe],
              "q_yaw": _ekf_q_yaw_by_airframe[airframe],
              "zeta": _ekf_zeta_by_airframe[airframe],
              "sigma_xy": _ekf_sigma_xy_by_airframe[airframe],
              "sigma_yaw": _ekf_sigma_yaw_by_airframe[airframe],
              "sigma_alpha_0": _ekf_sigma_alpha_0_by_airframe[airframe],
              "sigma_rate_0": _ekf_sigma_rate_0_by_airframe[airframe],
              "sigma_psi_p_0": _ekf_sigma_psi_p_0_by_airframe[airframe]}

    return tuning


##### transmitter control tuning #####
_stick_trim_by_airframe = {"AURELIA_X4": 1500, "IF1200": 1500,
                           "EKF_TEST": 1500}
STICK_TRIM = _stick_trim_by_airframe[AIRFRAME]
_stick_travel_by_airframe = {"AURELIA_X4": 397, "IF1200": 495,
                             "EKF_TEST": 495}
STICK_TRAVEL = _stick_travel_by_airframe[AIRFRAME]
_stick_dz_by_airframe = {"AURELIA_X4": 15, "IF1200": 15, "EKF_TEST": 15}
STICK_DZ = _stick_dz_by_airframe[AIRFRAME]
_a_max_by_airframe = {"AURELIA_X4": 2, "IF1200": 2, "EKF_TEST": 2}
MAX_STICK_ACCELERATION = _a_max_by_airframe[AIRFRAME]
_payload_v_xy_max_by_airframe = {"AURELIA_X4": 1.5, "IF1200": 1.5,
                                 "EKF_TEST": 1.5}  # m/s
_payload_v_z_max_by_airframe = {"AURELIA_X4": 0.5, "IF1200": 0.5,
                                "EKF_TEST": 0.5}  # m/s
PAYLOAD_V_XY_MAX = _payload_v_xy_max_by_airframe[AIRFRAME]
PAYLOAD_V_Z_MAX = _payload_v_z_max_by_airframe[AIRFRAME]
