import numpy as np

PI = np.pi
GRAVITY = 9.81  # [m/s^2]

##### drone parameters #####

MASS_DRONE = 2.55  # [kg]
J = np.diag([0.080, 0.080, 0.129])
# C_T = 8.7e-5
MAX_OMEGA = 1100
C_SIGMA_MAX = 70  # [N]

# geometry
ARM_LEN = 0.381
D = ARM_LEN / np.sqrt(2)
K_M = 0.02

###### payload parameters #####
MASS_TETHER = 0.68  # [kg]
MASS_PAYLOAD = 1.13  # [kg]
DISC_DIAMETER = 0.3  # [m]
DISC_WIDTH = 0.0025  # [m]
TETHER_LEN = 10.2  # [m]

# [m] tether pivot offset from drone CG in body frame
TETHER_PIVOT_OFFSET = np.zeros(3)


##### totals #####
MASS_PAYLOAD_EFF = MASS_PAYLOAD + MASS_TETHER
MASS_TOTAL = MASS_DRONE + MASS_PAYLOAD_EFF


##### camera tracking parameters #####

# settings
CAM_GAIN = 1
CAM_EXP_ABS = 8
CAM_PREVIEW_PORT = None
CAM_FPS = 48
CAM_STRIDE = 1


# camera offsets, referenced to the front of the drone
CAM_OFFSET_X = -0.34  # [m] (0.34 m port side)
CAM_OFFSET_Y = 0.12  # [m] (0.12 m in front of drone)
CAM_OFFSET_Z = -0.05  # [m]

# camera rotation


def Rx(a): c, s = np.cos(a), np.sin(a); return np.array(
    [[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a): c, s = np.cos(a), np.sin(a); return np.array(
    [[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a): c, s = np.cos(a), np.sin(a); return np.array(
    [[c, -s, 0], [s, c, 0], [0, 0, 1]])


CAM_ROLL = 0  # rad, about the nose axis (+Y, front)
CAM_PITCH = 0  # rad, about the lateral axis (+X, starboard)
CAM_YAW = PI/2  # rad, about vertical axis (+Z, top), right hand rule

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

##### transmitter control tuning #####
STICK_TRIM = 1500
STICK_TRAVEL = 397
STICK_DZ = 15
STICK_TAU = 2 # maximum acceleration is V_MAX / STICK_TAU
PAYLOAD_V_XY_MAX = 1.5 # m/s
PAYLOAD_V_Z_MAX = 0.5 # m/s
