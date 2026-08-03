"""
Camera pre-processing for the payload EKF.

Everything between the ArUco pose data and the numbers the filter and the
plots consume lives here: PnP marker poses -> payload board center and marker
row direction in the camera frame -> line of sight from the tether pivot ->
either the measurement vector the filter folds in, or the same frame written
as swing angles for a plot.

    z = [b_x, b_y, psi_P]

    b       unit line of sight from the tether pivot to the payload board
            center, camera frame. Only the first two components are carried,
            the third is redundant since ||b|| = 1
    psi_P   payload yaw, inertial frame, from the marker row direction

Rotation only: normalising the line of sight drops the tether length, so
nothing here needs to know how long the tether is.

Camera frame is the OpenCV convention config.CAM_R assumes:
+x right, +y DOWN, +z along the optical axis.
"""
import math
import numpy as np
import cv2

import Prm.config as config


# z = [b_x, b_y, psi_p]
MEAS_DIM = 3
IX_B_X, IX_B_Y, IX_PSI_MEAS = range(MEAS_DIM)

# signed distance from the board center along the marker row, +left [m]
MARKER_OFFSET = {config.LEFT_MARKER_ID: config.MARKER_CENTER_TO_CENTER_DIST,
                 config.CENTER_MARKER_ID: 0,
                 config.RIGHT_MARKER_ID: -config.MARKER_CENTER_TO_CENTER_DIST}

# camera <-> body, and the two body points the line of sight runs between
T_BC = config.CAM_R
T_CB = T_BC.T
t_BC_B = np.array([config.CAM_OFFSET_X,
                   config.CAM_OFFSET_Y,
                   config.CAM_OFFSET_Z])
l_B = np.array([0.0, 0.0, config.TETHER_PIVOT_OFFSET])


def get_payload_center_in_camera_frame(frame):
    """
    (payload board center, marker row direction) in the camera frame.

    Every detected marker gives its own estimate of both, from its PnP pose
    and its known offset along the row, and the estimates are averaged.
    Returns None when the frame holds no detection
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


def pivot_bearing(p_C):
    """
    Camera frame point -> UNIT line of sight to it FROM THE TETHER PIVOT,
    still in the camera frame.

    Shifts the origin camera -> body -> pivot, then normalises
    """
    p = np.asarray(p_C, float) + T_CB @ (t_BC_B - l_B)

    return p/np.linalg.norm(p)


def payload_yaw(mC, T_IB):
    """
    Payload yaw in the inertial frame from the marker row direction.

    atan2 already lands in (-pi, pi], so no wrapping is needed
    """
    mI = T_IB @ (T_BC @ np.asarray(mC, float))

    return math.atan2(mI[1], mI[0])


def alpha_from_q_I(q):
    """
    Returns (alpha_x, alpha_y) from any pivot->payload vector expressed in
    the inertial frame
    """
    q = np.asarray(q, dtype=float)
    q = q/np.linalg.norm(q)
    alpha_y = math.asin(np.clip(q[1], -1, 1))
    alpha_x = math.atan2(q[0], -q[2])

    return alpha_x, alpha_y


def measurement(frame, T_IB):
    """
    One camera frame -> z, the vector the filter folds in.

    Returns None when the frame holds no detection, which is the filter's cue
    to coast on the process model
    """
    detection = get_payload_center_in_camera_frame(frame)
    if detection is None:
        return None
    p_ctr_C, mC = detection

    b = pivot_bearing(p_ctr_C)

    z = np.zeros(MEAS_DIM)
    z[IX_B_X] = b[0]
    z[IX_B_Y] = b[1]
    z[IX_PSI_MEAS] = payload_yaw(mC, T_IB)

    return z


def swing_angles(frame, T_IB):
    """
    One camera frame -> (alpha_x, alpha_y, psi_P), or None with no detection.

    The same measurement in the states the filter carries rather than in the
    camera frame: what one frame on its own says the payload is doing. Used to
    seed the filter, and to plot the unfiltered measurement against it
    """
    detection = get_payload_center_in_camera_frame(frame)
    if detection is None:
        return None
    p_ctr_C, mC = detection

    b = pivot_bearing(p_ctr_C)
    alpha_x, alpha_y = alpha_from_q_I(T_IB @ (T_BC @ b))

    return alpha_x, alpha_y, payload_yaw(mC, T_IB)


def marker_bearings(frame):
    """
    {marker_id: unit line of sight from the pivot}, one per detected marker.

    The individual markers behind the board center, for plots that show which
    ones PnP had to work with
    """
    out = {}
    for marker in frame.dropna(subset=["marker_id"]).itertuples():
        out[int(marker.marker_id)] = pivot_bearing([marker.x,
                                                    marker.y,
                                                    marker.z])

    return out
