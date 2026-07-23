import numpy as np

PI = np.pi

MASS_DRONE = 5
MASS_TETHER = 0.680389 # [kg]
MASS_PAYLOAD = 1.13398 # [kg]
MASS_TOTAL = MASS_DRONE + MASS_PAYLOAD
GRAVITY = 9.81
TETHER_LEN = 8
# C_T = 8.7e-5
MAX_OMEGA = 1100

J = np.diag([0.09, 0.09, 0.17])

# geometry
ARM_LEN = 0.381
D = ARM_LEN / np.sqrt(2)
K_M = 0.02

# mission params (applied to the payload)
CRUISE_SPEED = 1
GRID_X_START = 0
GRID_Y_START = 0
GRID_X_END = 10
GRID_Y_END = 10
GRID_ALTITUDE = 15

##### camera tracking parameters #####

# camera offsets, referenced to the front of the drone
CAM_OFFSET_X = -0.34 # [m] (0.34 m port side)
CAM_OFFSET_Y = 0.12 # [m] (0.12 m in front of drone)
CAM_OFFSET_Z = -0.05 # [m]
CAM_ROLL = 0 # rad
CAM_PITCH = 0 # rad
CAM_YAW = -PI/2 # rad

# payload marker parameters
MARKER_SIZE = 0.17 # [m]
MARKER_CENTER_TO_CENTER_DIST = 0.505 # [m]
LEFT_MARKER_ID = 232
CENTER_MARKER_ID = 245
RIGHT_MARKER_ID = 233
