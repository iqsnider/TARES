import cv2
import json
import numpy as np
from ChArUco_board import ARUCO_DICT

MARKER_SIZE_m = 0.1356  # [m] length of black marker

# load camera parameters
with open("calibration.json") as f:
    calib = json.load(f)
mtx = np.array(calib["mtx"], dtype=np.float64)
dist = np.array(calib["dist"], dtype=np.float64)

# aruco detector
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

s = MARKER_SIZE_m/2
obj_points = np.array([[-s, s, 0],
                       [s, s, 0],
                       [s, -s, 0],
                       [-s, -s, 0]], dtype=np.float32)


def get_marker_poses(frame):
    # gray scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    poses = {}
    if ids is not None:
        for marker_corners, marker_id in zip(corners, ids.flatten()):
            ok, rvec, tvec = cv2.solvePnP(
                obj_points, marker_corners[0], mtx, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                poses[int(marker_id)] = (rvec, tvec)
    return corners, ids, poses


if __name__ == '__main__':

    # test run
    CALIB_W = int(round(2*mtx[0, 2]))
    CALIB_H = int(round(2*mtx[1, 2]))

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CALIB_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CALIB_H)

    ok, frame = cap.read()
    h, w = frame.shape[:2]
    print(f"requested {CALIB_W}x{CALIB_H}, got {w}x{h}")
    assert abs(w - CALIB_W) < 0.1 * \
        CALIB_W, "resolution mismatch, pose will be wrong"

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        corners, ids, poses = get_marker_poses(frame)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for marker_id, (rvec, tvec) in poses.items():
            cv2.drawFrameAxes(frame, mtx, dist, rvec,
                              tvec, MARKER_SIZE_m * 0.5)
            x, y, z = tvec.flatten()
            dist_m = float(np.linalg.norm(tvec))
            print(f"id {marker_id}: x={x:+.3f} y={y:+.3f} z={z:+.3f} m  "
                  f"range={dist_m:.3f} m")

        cv2.imshow("pose", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
