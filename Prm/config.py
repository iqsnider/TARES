import numpy as np

PI = np.pi
GRAVITY = 9.81  # [m/s^2]

# which physical airframe is currently mounted. Selects every per-airframe
# value below; only what's the same on both rigs stays a bare constant.
AIRFRAME = "IF1200"  # or "AURELIA_X4"

##### drone parameters #####

_mass_drone_by_airframe = {"AURELIA_X4": 2.55, "IF1200": 17}  # [kg]
MASS_DRONE = _mass_drone_by_airframe[AIRFRAME]

_j_by_airframe = {"AURELIA_X4": np.diag([0.080, 0.080, 0.129]),
                  "IF1200": np.diag([0.080, 0.080, 0.129])}
J = _j_by_airframe[AIRFRAME]
# C_T = 8.7e-5
_max_omega_by_airframe = {"AURELIA_X4": 1100, "IF1200": 1100}
MAX_OMEGA = _max_omega_by_airframe[AIRFRAME]

# geometry
_arm_len_by_airframe = {"AURELIA_X4": 0.381, "IF1200": 0.381}  # [m]
ARM_LEN = _arm_len_by_airframe[AIRFRAME]
D = ARM_LEN / np.sqrt(2)
_k_m_by_airframe = {"AURELIA_X4": 0.02, "IF1200": 0.02}
K_M = _k_m_by_airframe[AIRFRAME]

###### payload parameters #####
MASS_TETHER = 0.68  # [kg]
_mass_payload_by_airframe = {"AURELIA_X4": 1.13, "IF1200": 8.6}  # [kg]
MASS_PAYLOAD = _mass_payload_by_airframe[AIRFRAME]
DISC_DIAMETER = 0.3  # [m]
DISC_WIDTH = 0.0025  # [m]
_tether_len_by_airframe = {"AURELIA_X4": 10.2, "IF1200": 8.1}  # [m]
TETHER_LEN = _tether_len_by_airframe[AIRFRAME]

# [m] tether pivot offset from drone CG in body frame
TETHER_PIVOT_OFFSET = np.zeros(3)


##### totals #####
MASS_PAYLOAD_EFF = MASS_PAYLOAD + MASS_TETHER
MASS_TOTAL = MASS_DRONE + MASS_PAYLOAD_EFF


##### camera tracking parameters #####

# settings
CAM_GAIN = 1
_cam_exp_abs_by_airframe = {"AURELIA_X4": 8, "IF1200": 7}
CAM_EXP_ABS = _cam_exp_abs_by_airframe[AIRFRAME]
CAM_PREVIEW_PORT = None
CAM_FPS = 48
CAM_STRIDE = 1


# camera offsets, referenced to the front of the drone and CoG [m]
_cam_offset_x_by_airframe = {"AURELIA_X4": -0.34, "IF1200": -0.1}
_cam_offset_y_by_airframe = {"AURELIA_X4": 0.12, "IF1200": -0.154}
_cam_offset_z_by_airframe = {"AURELIA_X4": -0.05, "IF1200": -0.1}
CAM_OFFSET_X = _cam_offset_x_by_airframe[AIRFRAME]
CAM_OFFSET_Y = _cam_offset_y_by_airframe[AIRFRAME]
CAM_OFFSET_Z = _cam_offset_z_by_airframe[AIRFRAME]

# camera rotation


def Rx(a): c, s = np.cos(a), np.sin(a); return np.array(
    [[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a): c, s = np.cos(a), np.sin(a); return np.array(
    [[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a): c, s = np.cos(a), np.sin(a); return np.array(
    [[c, -s, 0], [s, c, 0], [0, 0, 1]])


_cam_roll_by_airframe = {"AURELIA_X4": 0, "IF1200": 0}
_cam_pitch_by_airframe = {"AURELIA_X4": 0, "IF1200": 0}
# rad, about the nose axis (+Y, front)
CAM_ROLL = _cam_roll_by_airframe[AIRFRAME]
# rad, about the lateral axis (+X, starboard)
CAM_PITCH = _cam_pitch_by_airframe[AIRFRAME]
# rad, about vertical axis (+Z, top), right hand rule (0 when up on the camera frame is in the drone nose direction)
_cam_yaw_by_airframe = {"AURELIA_X4": PI/2, "IF1200": PI}
CAM_YAW = _cam_yaw_by_airframe[AIRFRAME]

# camera at zero rpy stares straight down, top of the image toward the nose
R_CAM_DOWN = np.diag([1, -1, -1])
R_MOUNT = Rz(CAM_YAW)@Rx(CAM_PITCH)@Ry(CAM_ROLL)
CAM_R = R_MOUNT@R_CAM_DOWN


# payload marker parameters
MARKER_EDGE_LEN = 0.17  # [m]
MARKER_CENTER_TO_CENTER_DIST = 0.505  # [m]
LEFT_MARKER_ID = 30
CENTER_MARKER_ID = 245
RIGHT_MARKER_ID = 233


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
_lqr_drone_q_pos_xy_by_airframe = {"AURELIA_X4": 1, "IF1200": 1}
_lqr_drone_q_pos_z_by_airframe = {"AURELIA_X4": 1, "IF1200": 1}
_lqr_drone_q_vel_xy_by_airframe = {"AURELIA_X4": 1, "IF1200": 1}
_lqr_drone_q_vel_z_by_airframe = {"AURELIA_X4": 1, "IF1200": 1}
_lqr_drone_r_acc_by_airframe = {"AURELIA_X4": 1, "IF1200": 1}
LQR_DRONE_Q_POS_XY = _lqr_drone_q_pos_xy_by_airframe[AIRFRAME]
LQR_DRONE_Q_POS_Z = _lqr_drone_q_pos_z_by_airframe[AIRFRAME]
LQR_DRONE_Q_VEL_XY = _lqr_drone_q_vel_xy_by_airframe[AIRFRAME]
LQR_DRONE_Q_VEL_Z = _lqr_drone_q_vel_z_by_airframe[AIRFRAME]
LQR_DRONE_R_ACC = _lqr_drone_r_acc_by_airframe[AIRFRAME]

# payload outer loop (OuterLoopPayloadLQR)
_lqr_payload_w_pos_xy_by_airframe = {"AURELIA_X4": 1, "IF1200": (1/1.2)**2}
_lqr_payload_w_pos_z_by_airframe = {"AURELIA_X4": 1, "IF1200": (1/1.2)**2}
_lqr_payload_tuning_const_by_airframe = {
    "AURELIA_X4": 1/1**2, "IF1200": 1/1**2}
LQR_PAYLOAD_W_POS_XY = _lqr_payload_w_pos_xy_by_airframe[AIRFRAME]
LQR_PAYLOAD_W_POS_Z = _lqr_payload_w_pos_z_by_airframe[AIRFRAME]
LQR_PAYLOAD_TUNING_CONST = _lqr_payload_tuning_const_by_airframe[AIRFRAME]

# payload outer loop with integral action (OuterLoopPayloadLQI)
_lqi_payload_w_pos_xy_by_airframe = {"AURELIA_X4": 1, "IF1200": (1/1.5)**2}
_lqi_payload_w_pos_z_by_airframe = {"AURELIA_X4": 1, "IF1200": (1/1.5)**2}
_lqi_payload_w_int_xy_by_airframe = {"AURELIA_X4": 1/4, "IF1200": (1/8)**2}
_lqi_payload_w_int_z_by_airframe = {"AURELIA_X4": 1/4, "IF1200": (1/8)**2}
_lqi_payload_tuning_const_by_airframe = {
    "AURELIA_X4": (1/1)**2, "IF1200": (1/1)**2}
LQI_PAYLOAD_W_POS_XY = _lqi_payload_w_pos_xy_by_airframe[AIRFRAME]
LQI_PAYLOAD_W_POS_Z = _lqi_payload_w_pos_z_by_airframe[AIRFRAME]
LQI_PAYLOAD_W_INT_XY = _lqi_payload_w_int_xy_by_airframe[AIRFRAME]
LQI_PAYLOAD_W_INT_Z = _lqi_payload_w_int_z_by_airframe[AIRFRAME]
LQI_PAYLOAD_TUNING_CONST = _lqi_payload_tuning_const_by_airframe[AIRFRAME]
# error norm [m] below which the integrator is allowed to accumulate
_lqi_payload_e_band_by_airframe = {"AURELIA_X4": 5, "IF1200": 5}
LQI_PAYLOAD_E_BAND = _lqi_payload_e_band_by_airframe[AIRFRAME]
# cap on the integral term's own contribution to u [m/s^2], so it cannot
# run away even while accumulating. Sized per-airframe from each rig's own
# commanded-acceleration logs (p95 of a clean closed-loop flight).
_lqi_payload_u_i_max_by_airframe = {"AURELIA_X4": 1.5, "IF1200": 2}
LQI_PAYLOAD_U_I_MAX = _lqi_payload_u_i_max_by_airframe[AIRFRAME]

##### transmitter control tuning #####
_stick_trim_by_airframe = {"AURELIA_X4": 1500, "IF1200": 1500}
STICK_TRIM = _stick_trim_by_airframe[AIRFRAME]
_stick_travel_by_airframe = {"AURELIA_X4": 397, "IF1200": 495}
STICK_TRAVEL = _stick_travel_by_airframe[AIRFRAME]
_stick_dz_by_airframe = {"AURELIA_X4": 15, "IF1200": 15}
STICK_DZ = _stick_dz_by_airframe[AIRFRAME]
# maximum acceleration is V_MAX / STICK_TAU
_stick_tau_by_airframe = {"AURELIA_X4": 2, "IF1200": 2}
STICK_TAU = _stick_tau_by_airframe[AIRFRAME]
_payload_v_xy_max_by_airframe = {"AURELIA_X4": 1.5, "IF1200": 1.5}  # m/s
_payload_v_z_max_by_airframe = {"AURELIA_X4": 0.5, "IF1200": 0.5}  # m/s
PAYLOAD_V_XY_MAX = _payload_v_xy_max_by_airframe[AIRFRAME]
PAYLOAD_V_Z_MAX = _payload_v_z_max_by_airframe[AIRFRAME]
