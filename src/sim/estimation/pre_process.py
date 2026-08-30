import dataclasses
import math
import numpy as np
import cv2

import Prm.config as config


MEAS_DIM = 3
IX_B_X, IX_B_Y, IX_PSI_MEAS = range(MEAS_DIM)


@dataclasses.dataclass(frozen=True)
class Geometry:
    """
    Camera/marker-board geometry for a swing-angle computation.

    Prm/config.py changes between experiments, so build this from a
    session's config_snapshot.json instead when analyzing old data.
    """
    marker_offset: dict
    T_BC: np.ndarray
    t_BC_B: np.ndarray
    l_B: np.ndarray

    @property
    def T_CB(self):
        t_cb = self.T_BC.T
        return t_cb

    @classmethod
    def from_config(cls, cfg=config):
        geom = cls(
            marker_offset={cfg.LEFT_MARKER_ID: cfg.MARKER_CENTER_TO_CENTER_DIST,
                          cfg.CENTER_MARKER_ID: 0,
                          cfg.RIGHT_MARKER_ID: -cfg.MARKER_CENTER_TO_CENTER_DIST},
            T_BC=np.asarray(cfg.CAM_R, dtype=float),
            t_BC_B=np.array([cfg.CAM_OFFSET_X, cfg.CAM_OFFSET_Y, cfg.CAM_OFFSET_Z]),
            l_B=np.asarray(cfg.TETHER_PIVOT_OFFSET, dtype=float))
        return geom

    @classmethod
    def from_snapshot(cls, snap):
        """From a session's config_snapshot.json (see catalog.Session.config)."""
        geom = cls(
            marker_offset={snap["LEFT_MARKER_ID"]: snap["MARKER_CENTER_TO_CENTER_DIST"],
                          snap["CENTER_MARKER_ID"]: 0,
                          snap["RIGHT_MARKER_ID"]: -snap["MARKER_CENTER_TO_CENTER_DIST"]},
            T_BC=np.array(snap["CAM_R"], dtype=float),
            t_BC_B=np.array([snap["CAM_OFFSET_X"], snap["CAM_OFFSET_Y"],
                             snap["CAM_OFFSET_Z"]]),
            l_B=np.array(snap["TETHER_PIVOT_OFFSET"], dtype=float))
        return geom


# default for live call sites that don't pass their own geometry
DEFAULT_GEOMETRY = Geometry.from_config()


def center_from_records(records, geom=DEFAULT_GEOMETRY):
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

        offset = geom.marker_offset[int(marker_id)]

        center_estimates.append(tC + offset*mC)

    if not center_estimates:
        return None

    approx_center = np.mean(center_estimates, axis=0)
    approx_mC = np.mean(mC_estimates, axis=0)

    return approx_center, approx_mC


def get_payload_center_in_camera_frame(frame, geom=DEFAULT_GEOMETRY):
    """
    Finds the payload center in the camera frame (offline: a poses.csv frame)
    """
    records = [(marker.marker_id,
                (marker.rx, marker.ry, marker.rz),
                (marker.x, marker.y, marker.z))
               for marker in frame.dropna(subset=["marker_id"]).itertuples()]

    center = center_from_records(records, geom=geom)
    return center


def pivot_to_payload(pC_ctr, geom=DEFAULT_GEOMETRY):
    """
    Vector from the tether pivot to the payload center, camera frame [m]
    """
    pC = np.asarray(pC_ctr, float) + geom.T_CB @ (geom.t_BC_B - geom.l_B)

    return pC


def shift_origin(pC_ctr, geom=DEFAULT_GEOMETRY):
    """
    Shift the origin from the camera optical center to the tether pivot point
    """
    pC = pivot_to_payload(pC_ctr, geom=geom)

    normalized_pC = pC/np.linalg.norm(pC)

    return normalized_pC


def payload_yaw(mC, T_IB, geom=DEFAULT_GEOMETRY):
    """
    Returns the payload yaw in the inertial frame from the averaged payload center mC in the camera frame
    """
    mI = T_IB @ (geom.T_BC @ np.asarray(mC, float))

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


def _build_z(detection, T_IB, geom=DEFAULT_GEOMETRY):
    """
    Assemble the EKF measurement vector from a (center, mC) detection
    """
    p_ctr_C, mC = detection

    b = shift_origin(p_ctr_C, geom=geom)

    z = np.zeros(MEAS_DIM)
    z[IX_B_X] = b[0]
    z[IX_B_Y] = b[1]
    z[IX_PSI_MEAS] = payload_yaw(mC, T_IB, geom=geom)

    return z


def measurement(frame, T_IB, geom=DEFAULT_GEOMETRY):
    """
    Returns the measurement needed for the EKF from a given frame and drone attitude
    """
    detection = get_payload_center_in_camera_frame(frame, geom=geom)
    if detection is None:
        return None

    z = _build_z(detection, T_IB, geom=geom)
    return z


def measurement_from_poses(poses, T_IB, geom=DEFAULT_GEOMETRY):
    """
    Live equivalent of measurement(), taking the recorder's pose dict.

    poses is {marker_id: (rvec, tvec)} straight off MarkerPoseRecorder. Returns
    None when nothing was detected this frame, which the EKF treats as
    predict-only.
    """
    if not poses:
        return None

    detection = center_from_records(
        [(mid, rvec, tvec) for mid, (rvec, tvec) in poses.items()], geom=geom)
    if detection is None:
        return None

    z = _build_z(detection, T_IB, geom=geom)
    return z


def range_from_poses(poses, geom=DEFAULT_GEOMETRY):
    """
    Distance from the tether pivot to the payload center [m], or None.

    The measurement the EKF throws away when it normalizes the bearing, which
    is what a tether length is: hang the payload still and read this.
    """
    if not poses:
        return None

    detection = center_from_records(
        [(mid, rvec, tvec) for mid, (rvec, tvec) in poses.items()], geom=geom)
    if detection is None:
        return None

    r = float(np.linalg.norm(pivot_to_payload(detection[0], geom=geom)))

    return r


def measurement_from_circle(p_C, geom=DEFAULT_GEOMETRY):
    """
    EKF measurement from a color ring detection, which is bearing only.

    p_C is the ring center in the camera frame, straight off
    ColorCircleRecorder.detect. A circle is symmetric, so there is no payload
    yaw in it and the measurement is two long instead of three.
    """
    if p_C is None:
        return None

    b = shift_origin(p_C, geom=geom)

    z = np.array([b[IX_B_X], b[IX_B_Y]])

    return z


def swing_angles(frame, T_IB, geom=DEFAULT_GEOMETRY):
    """
    One camera frame (alpha_x, alpha_y, psi_P), or None with no detection
    """
    detection = get_payload_center_in_camera_frame(frame, geom=geom)
    if detection is None:
        return None
    p_ctr_C, mC = detection

    b = shift_origin(p_ctr_C, geom=geom)
    alpha_x, alpha_y = alpha_from_q_I(T_IB @ (geom.T_BC @ b))
    psi_p = payload_yaw(mC, T_IB, geom=geom)

    return alpha_x, alpha_y, psi_p


def swing_angles_from_poses(poses, T_IB, geom=DEFAULT_GEOMETRY):
    """
    Live equivalent of swing_angles, taking the recorder's pose dict.

    poses is {marker_id: (rvec, tvec)} straight off MarkerPoseRecorder, the
    same shape measurement_from_poses reads.
    """
    if not poses:
        return None

    detection = center_from_records(
        [(mid, rvec, tvec) for mid, (rvec, tvec) in poses.items()], geom=geom)
    if detection is None:
        return None
    p_ctr_C, mC = detection

    b = shift_origin(p_ctr_C, geom=geom)
    alpha_x, alpha_y = alpha_from_q_I(T_IB @ (geom.T_BC @ b))
    psi_p = payload_yaw(mC, T_IB, geom=geom)

    return alpha_x, alpha_y, psi_p


def swing_angles_from_circle(p_C, T_IB, geom=DEFAULT_GEOMETRY):
    """
    One color detection (alpha_x, alpha_y, psi_P), or None with no detection.

    Shaped like swing_angles so a caller can seed a filter from either
    tracker, but a ring is symmetric and carries no orientation, so the yaw
    it reports is 0 rather than measured.
    """
    if p_C is None:
        return None

    b = shift_origin(p_C, geom=geom)
    alpha_x, alpha_y = alpha_from_q_I(T_IB @ (geom.T_BC @ b))

    return alpha_x, alpha_y, 0


def marker_bearings(frame, geom=DEFAULT_GEOMETRY):
    """
    The individual markers behind the board center
    """
    out = {}
    for marker in frame.dropna(subset=["marker_id"]).itertuples():
        out[int(marker.marker_id)] = shift_origin([marker.x,
                                                   marker.y,
                                                   marker.z], geom=geom)

    return out
