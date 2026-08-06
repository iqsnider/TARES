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


def center_from_records(records):
    """
    Payload board center and marker x-axis in the camera frame.

    records is an iterable of (marker_id, rvec, tvec). This is the shared core
    for both the offline CSV path and the live control-loop path, so the two
    cannot drift apart.
    """
    center_estimates = []
    mC_estimates = []

    for marker_id, rvec, tvec in records:
        # rotation matrix from the marker's rodrigues vector
        T_CM, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3))

        # position
        tC = np.asarray(tvec, dtype=float).reshape(3)

        # marker x axis
        mC = T_CM[:, 0]
        mC_estimates.append(mC)

        offset = MARKER_OFFSET[int(marker_id)]

        center_estimates.append(tC + offset*mC)

    if not center_estimates:
        return None

    approx_center = np.mean(center_estimates, axis=0)
    approx_mC = np.mean(mC_estimates, axis=0)

    return approx_center, approx_mC


def get_payload_center_in_camera_frame(frame):
    """
    Finds the payload center in the camera frame (offline: a poses.csv frame)
    """
    records = [(marker.marker_id,
                (marker.rx, marker.ry, marker.rz),
                (marker.x, marker.y, marker.z))
               for marker in frame.dropna(subset=["marker_id"]).itertuples()]

    return center_from_records(records)


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

    z_psi_p = math.atan2(mI[1], mI[0])

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


def _build_z(detection, T_IB):
    """
    Assemble the EKF measurement vector from a (center, mC) detection
    """
    p_ctr_C, mC = detection

    b = shift_origin(p_ctr_C)

    z = np.zeros(MEAS_DIM)
    z[IX_B_X] = b[0]
    z[IX_B_Y] = b[1]
    z[IX_PSI_MEAS] = payload_yaw(mC, T_IB)

    return z


def measurement(frame, T_IB):
    """
    Returns the measurement needed for the EKF from a given frame and drone attitude
    """
    detection = get_payload_center_in_camera_frame(frame)
    if detection is None:
        return None

    return _build_z(detection, T_IB)


def measurement_from_poses(poses, T_IB):
    """
    Live equivalent of measurement(), taking the recorder's pose dict.

    poses is {marker_id: (rvec, tvec)} straight off MarkerPoseRecorder. Returns
    None when nothing was detected this frame, which the EKF treats as
    predict-only.
    """
    if not poses:
        return None

    detection = center_from_records(
        [(mid, rvec, tvec) for mid, (rvec, tvec) in poses.items()])
    if detection is None:
        return None

    return _build_z(detection, T_IB)


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
