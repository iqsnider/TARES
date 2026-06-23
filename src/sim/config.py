import numpy as np

MASS_DRONE = 5
MASS_PAYLOAD = 0.5
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
GRID_X_END = 100
GRID_Y_END = 0
GRID_ALTITUDE = 1
