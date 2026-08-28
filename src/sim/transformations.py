"""
Transformations library

I   inertial (ENU: east, north, up)
B   body (ENU ordered: starboard, nose, up)
C   camera
P   payload

Functions are named T_AB: rotation to frame A from frame B, so
T_AB @ v_B == v_A.
"""
import numpy as np


# camera at zero mount rpy stares straight down, image top toward the nose
_R_CAM_DOWN = np.diag([1, -1, -1])


def Rx(a):
    """
    Elementary rotation about x
    """
    c, s = np.cos(a), np.sin(a)
    R = np.array([[1, 0, 0],
                 [0, c, -s],
                 [0, s, c]])

    return R


def Ry(a):
    """
    Elementary rotation about y
    """
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, 0, s],
                 [0, 1, 0],
                 [-s, 0, c]])

    return R


def Rz(a):
    """
    Elementary rotation about z
    """
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s, 0],
                 [s, c, 0],
                 [0, 0, 1]])

    return R


def T_ENU_from_NED():
    """
    Rotation to ENU from NED
    """
    T = np.array([[0, 1, 0],
                 [1, 0, 0],
                 [0, 0, -1]])

    return T


def T_IB(phi, theta, psi):
    """
    Rotation to the inertial frame from the body frame
    """
    sp, cp = np.sin(phi), np.cos(phi)
    st, ct = np.sin(theta), np.cos(theta)
    sy, cy = np.sin(psi), np.cos(psi)

    T = np.array([[ct*cy, sp*st*cy - cp*sy, cp*st*cy + sp*sy],
                 [ct*sy, sp*st*sy + cp*cy, cp*st*sy - sp*cy],
                 [-st, sp*ct, cp*ct]])

    return T


def T_BC(cam_roll, cam_pitch, cam_yaw):
    """
    Rotation to the body frame from the camera frame
    """
    T = Rz(cam_yaw) @ Rx(cam_pitch) @ Ry(cam_roll) @ _R_CAM_DOWN

    return T
