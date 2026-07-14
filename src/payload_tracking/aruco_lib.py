import cv2
import csv
import json
import time
import numpy as np
from payload_tracking.camera_calibration.ChArUco_board import ARUCO_DICT

import os


class MarkerPoseRecorder:
    def __init__(self,
                 calibration_path="~/TARES_SITL/src/payload_tracking/camera_calibration/calibration.json",
                 marker_size_m=0.14,
                 video_out="recording.avi",
                 csv_out="poses.csv",
                 fps=60):
        self.marker_size_m = marker_size_m
        self.video_out = os.path.expanduser(video_out)
        self.csv_out = os.path.expanduser(csv_out)
        self.fps = fps

        # load camera parameters
        with open(os.path.expanduser(calibration_path)) as f:
            calib = json.load(f)
        self.mtx = np.array(calib["mtx"], dtype=np.float64)
        self.dist = np.array(calib["dist"], dtype=np.float64)

        # aruco detector
        dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
        self.detector = cv2.aruco.ArucoDetector(
            dictionary, cv2.aruco.DetectorParameters())

        # marker object points (centered square)
        s = marker_size_m / 2
        self.obj_points = np.array([[-s, s, 0],
                                    [s, s, 0],
                                    [s, -s, 0],
                                    [-s, -s, 0]], dtype=np.float32)

        # calibration resolution
        self.calib_w = int(round(2 * self.mtx[0, 2]))
        self.calib_h = int(round(2 * self.mtx[1, 2]))

        # handles created when recording starts
        self.cap = None
        self.writer = None
        self.csv_file = None
        self.csv_writer = None
        self.frame_idx = 0
        self._t0 = None

    def get_marker_poses(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        poses = {}
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                ok, rvec, tvec = cv2.solvePnP(
                    self.obj_points, marker_corners[0], self.mtx, self.dist,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    poses[int(marker_id)] = (rvec, tvec)
        return corners, ids, poses

    def open(self, camera_index=0):
        """Open the camera and the output video/CSV files."""
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.calib_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.calib_h)

        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("could not read from camera")
        h, w = frame.shape[:2]
        print(f"requested {self.calib_w}x{self.calib_h}, got {w}x{h}")
        assert abs(w - self.calib_w) < 0.1 * self.calib_w, \
            "resolution mismatch, pose will be wrong"

        self.writer = cv2.VideoWriter(
            self.video_out, cv2.VideoWriter_fourcc(*"MJPG"),
            self.fps, (w, h))

        self.csv_file = open(self.csv_out, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            ["frame", "time_s", "marker_id",
             "rx", "ry", "rz", "x", "y", "z", "range_m"])

        self.frame_idx = 0
        self._t0 = time.time()
        return self

    def process_frame(self, frame):
        """Detect markers, annotate the frame, write video + CSV rows.

        Returns the poses dict for this frame.
        """
        corners, ids, poses = self.get_marker_poses(frame)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        t = time.time() - self._t0
        if poses:
            for marker_id, (rvec, tvec) in poses.items():
                cv2.drawFrameAxes(frame, self.mtx, self.dist, rvec,
                                  tvec, self.marker_size_m * 0.5)
                rx, ry, rz = rvec.flatten()
                x, y, z = tvec.flatten()
                dist_m = float(np.linalg.norm(tvec))
                print(f"id {marker_id}: x={x:+.3f} y={y:+.3f} z={z:+.3f} m  "
                      f"range={dist_m:.3f} m")
                self.csv_writer.writerow(
                    [self.frame_idx, f"{t:.4f}", marker_id,
                     f"{rx:.6f}", f"{ry:.6f}", f"{rz:.6f}",
                     f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{dist_m:.6f}"])
        else:
            nan = float("nan")
            print("no marker detected")
            self.csv_writer.writerow(
                [self.frame_idx, f"{t:.4f}", nan,
                 nan, nan, nan, nan, nan, nan, nan])

        self.writer.write(frame)
        self.frame_idx += 1
        return poses

    def run(self, camera_index=0):
        """Open everything and record until Ctrl+C or camera failure."""
        self.open(camera_index)
        print("recording... press Ctrl+C to stop")
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break
                self.process_frame(frame)
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def close(self):
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        if self.csv_file is not None:
            self.csv_file.close()
        print(f"saved {self.frame_idx} frames to {self.video_out} "
              f"and poses to {self.csv_out}")

    # context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == '__main__':
    recorder = MarkerPoseRecorder(
        calibration_path="calibration.json",
        marker_size_m=0.1356,
        video_out="recording.avi",
        csv_out="poses.csv",
        fps=60)
    recorder.run()
