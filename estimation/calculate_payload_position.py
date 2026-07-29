"""
library of helper functions for:

payload pose/position csv -> payload position in payload frame -> camera reference frame
                                                            |
                                                            V
camera position on drone -> payload position in drone reference frame
                                                |
                                                V
drone position/attitude csv -> payload position in inertial reference frame
"""
import numpy as np
import csv
import pandas as pd
import cv2

import sim.config as config

MARKER_OFFSET = {config.LEFT_MARKER_ID: config.MARKER_CENTER_TO_CENTER_DIST,
          config.CENTER_MARKER_ID: 0,
          config.RIGHT_MARKER_ID: -config.MARKER_CENTER_TO_CENTER_DIST}


def get_payload_center_in_camera_frame(frame):
    """
    Computes the payload position in the camera reference frame from the available data in a frame (picture)
    """
    # array of center estimates
    center_estimates = []

    # separate the markers detected in each frame
    markers_detected = frame.dropna(subset=["marker_id"]).itertuples()
    for marker in markers_detected:
        # get the rotation matrix
        R, _ = cv2.Rodrigues(np.array([marker.rx, marker.ry, marker.rz]))

        # position
        t = np.array([marker.x, marker.y, marker.z])

        offset = MARKER_OFFSET[int(marker.marker_id)]

        center_estimate = t + offset*R[:, 0]
        center_estimates.append(center_estimate)

    if not center_estimates:
        return None

    approx_center = np.mean(center_estimates, axis=0)

    return approx_center


def get_payload_center_in_drone_frame(payload_center_in_camera_frame):
    """
    Computes the payload position in the drone reference frame from the payload center in the camera frame and the given camera offset
    """
    if payload_center_in_camera_frame is None:
        return None

    # camera translation
    t = np.array([config.CAM_OFFSET_X, config.CAM_OFFSET_Y, config.CAM_OFFSET_Z])

    # camera rotation
    rot = config.CAM_R @ np.asarray(payload_center_in_camera_frame)

    # position in drone frame
    payload_center_in_drone_frame = rot + t

    return payload_center_in_drone_frame


def get_payload_center_in_inertial_frame(payload_center_in_drone_frame, drone_position_attitude):
    """
    Given the drone attitude and position data at the approriate timestamp,
    computes the payload position in ENU in the inertial frame from the payload center in the drone frame
    """
    if payload_center_in_drone_frame is None:
        return None

    P_NED = np.array([[0, 1, 0],
                      [1, 0, 0],
                      [0, 0, -1]])

    R_ned = (config.Rz(drone_position_attitude.drone_yaw)
             @ config.Ry(drone_position_attitude.drone_pitch)
             @ config.Rx(drone_position_attitude.drone_roll))
    R = P_NED @ R_ned @ P_NED

    # drone position in ENU
    p_drone_inertial = np.array([drone_position_attitude.drone_px_meas,
                        drone_position_attitude.drone_py_meas,
                        drone_position_attitude.drone_pz_meas])
    
    p_payload_inertial = R @ np.asarray(payload_center_in_drone_frame) + p_drone_inertial

    return p_payload_inertial


def get_attitude_at(drone_df, time_s, time_offset=0):
    """
    Nearest flight-log row to a camera timestamp
    """
    i = (drone_df.cur_time - (time_s - time_offset)).abs().idxmin()
    return drone_df.loc[i]


def get_payload_position_ENU(frame, drone_position_attitude):
    """
    Computes the payload position in the inertial frame ENU
    """
    payload_center_in_camera_frame = get_payload_center_in_camera_frame(frame)
    payload_center_in_drone_frame = get_payload_center_in_drone_frame(payload_center_in_camera_frame)
    payload_center_in_inertial_frame = get_payload_center_in_inertial_frame(payload_center_in_drone_frame, drone_position_attitude)

    return payload_center_in_inertial_frame


def get_payload_ENU_from_data(payload_pose_file, flight_data_file, time_offset=0):
    """
    Returns a dataframe with the same size as the flight data file so that the two can be graphed together later
    """
    payload_df = pd.read_csv(payload_pose_file)
    drone_df = pd.read_csv(flight_data_file).reset_index(drop=True)

    out = pd.DataFrame(np.nan, index=drone_df.index, columns=["payload_e", "payload_n", "payload_u",
                                                              "n_markers", "cam_time_s"])
    out["cur_time"] = drone_df.cur_time

    flight_time = drone_df.cur_time.to_numpy()

    for _, frame in payload_df.groupby("frame"):
        cam_time = frame.time_s.iloc[0]

        # nearest flight-log row to this camera frame, if one is close enough
        dt = np.abs(flight_time - (cam_time - time_offset))
        i = int(dt.argmin())
        if dt[i] > 1/config.CONTROL_FREQUENCY:
            continue

        payload_enu = get_payload_position_ENU(frame, drone_df.iloc[i])
        if payload_enu is None:
            continue

        out.iloc[i, 0:3] = payload_enu
        out.iloc[i, 3] = frame["marker_id"].notna().sum()
        out.iloc[i, 4] = cam_time

    return out






if __name__ == "__main__":
    payload_pose_file = "~/TARES_SITL/data/test_07232026/all_data_07232026/step_test_with_camera_20260723_114555/poses.csv"
    flight_data_file = "~/TARES_SITL/data/test_07232026/all_data_07232026/flight_20260723_114556.csv"
    payload_df = pd.read_csv(payload_pose_file)
    drone_df = pd.read_csv(flight_data_file)

    example_frame = payload_df[payload_df.frame == 348]
    example_position_attitude = get_attitude_at(drone_df, example_frame.time_s.iloc[0], time_offset=1)

    payload_center_in_camera_frame = get_payload_center_in_camera_frame(example_frame)
    payload_center_in_drone_frame = get_payload_center_in_drone_frame(payload_center_in_camera_frame)
    payload_center_in_inertial_frame = get_payload_center_in_inertial_frame(payload_center_in_drone_frame, example_position_attitude)

    # test out data
    payload_enu_data = get_payload_ENU_from_data(payload_pose_file, flight_data_file, time_offset=1)
    print(payload_enu_data)

