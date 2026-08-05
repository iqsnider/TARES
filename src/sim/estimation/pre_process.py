import math
import numpy as np
import cv2

import Prm.config as config


MEAS_DIM = 3
IX_B_X, IX_B_Y, IX_PSI_MEAS = range(MEAS_DIM)

MARKER_OFFSET = {config.LEFT_MARKER_ID: config.MARKER_CENTER_TO_CENTER_DIST,
                 config.CENTER_MARKER_ID: 0,
                 config.RIGHT_MARKER_ID: -config.MARKER_CENTER_TO_CENTER_DIST}

T_BC = config.CAM_R
T_CB = T_BC.T
t_BC_B = np.array([config.CAM_OFFSET_X,
                   config.CAM_OFFSET_Y,
                   config.CAM_OFFSET_Z])
l_B = config.TETHER_PIVOT_OFFSET


def get_payload_center_in_camera_frame(frame):
    """
    Finds the payload center in the camera frame
    """
    center_estimates = []
    mC_estimates = []

    # separate the markers detected in each frame
    markers_detected = frame.dropna(subset=["marker_id"]).itertuples()
    for marker in markers_detected:
        # get the rotation matrix
        T_CM, _ = cv2.Rodrigues(np.array([marker.rx, marker.ry, marker.rz]))

        # position
        tC = np.array([marker.x, marker.y, marker.z])

        # marker x axis
        mC = T_CM[:, 0]
        mC_estimates.append(mC)

        offset = MARKER_OFFSET[int(marker.marker_id)]

        center_estimate = tC + offset*mC
        center_estimates.append(center_estimate)

    if not center_estimates:
        return None

    approx_center = np.mean(center_estimates, axis=0)
    approx_mC = np.mean(mC_estimates, axis=0)

    return approx_center, approx_mC


def shift_origin(pC_ctr):
    """
    Shift the origin from the camera optical center to the tether pivot point
    """
    pC = np.asarray(pC_ctr, float) + T_CB @ (t_BC_B - l_B)

    normalized_pC = pC/np.linalg.norm(pC)

    return normalized_pC


def payload_yaw(mC, T_IB):
    """
    Returns the payload yaw in the inertial frame from the averaged payload center mC in the camera frame
    """
    mI = T_IB @ (T_BC @ np.asarray(mC, float))

    z_psi_p = math.atan2(mI[1],mI[0])

    return z_psi_p


def alpha_from_q_I(q):
    """
    returns alpha_x and alpha_y in the inertial frame
    """
    q = np.asarray(q, dtype=float)
    q = q/np.linalg.norm(q)
    alpha_y = math.asin(np.clip(q[1], -1, 1))
    alpha_x = math.atan2(q[0], -q[2])

    return alpha_x, alpha_y


def measurement(frame, T_IB):
    """
    Returns the measurement needed for the EKF from a given frame and drone attitude
    """
    detection = get_payload_center_in_camera_frame(frame)
    if detection is None:
        return None
    p_ctr_C, mC = detection

    b = shift_origin(p_ctr_C)

    z = np.zeros(MEAS_DIM)
    z[IX_B_X] = b[0]
    z[IX_B_Y] = b[1]
    z[IX_PSI_MEAS] = payload_yaw(mC, T_IB)

    return z


def swing_angles(frame, T_IB):
    """
    One camera frame (alpha_x, alpha_y, psi_P), or None with no detection
    """
    detection = get_payload_center_in_camera_frame(frame)
    if detection is None:
        return None
    p_ctr_C, mC = detection

    b = shift_origin(p_ctr_C)
    alpha_x, alpha_y = alpha_from_q_I(T_IB @ (T_BC @ b))

    return alpha_x, alpha_y, payload_yaw(mC, T_IB)


def marker_bearings(frame):
    """
    The individual markers behind the board center
    """
    out = {}
    for marker in frame.dropna(subset=["marker_id"]).itertuples():
        out[int(marker.marker_id)] = shift_origin([marker.x,
                                                    marker.y,
                                                    marker.z])

    return out
