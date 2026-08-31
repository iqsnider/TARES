import csv
import json
from collections import deque
import os
import shutil
import signal
import subprocess
import time

import cv2
import numpy as np

from payload_tracking.aruco_lib import (AUTO, _AsyncWriter, _FrameGrabber,
                                        _MJPEGPreview, apply_camera_controls)

NAN = float("nan")

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CALIB = os.path.join(_HERE, "camera_calibration/calibration.json")


class ColorCircleRecorder:
    def __init__(self,
                 calibration_path=_DEFAULT_CALIB,
                 circle_diameter=0.3,  # outer diameter of the ring
                 band=0.03,  # width of the colored band
                 hue=110,  # 0-179
                 hue_width=15,
                 sat_min=90,
                 val_min=50,
                 min_area_px=150,
                 min_coverage_deg=200,  # the minimum angle that the blob must cover
                 expected_range=None,  # the tether length, if known
                 range_tol=0.35,  # tolerance around the tether length [m]
                 morph_px=3,  # kernel for closing the gap
                 video_out="recording.avi",
                 csv_out="circles.csv",
                 capture_fps=48,
                 frame_stride=1,
                 width=2304,
                 height=1536,
                 pixel_format="MJPG",
                 exposure_abs=2,  # units of 100 microseconds
                 gain=1,
                 print_every=30,
                 write_queue_size=16,
                 preview_port=None,
                 preview_fps=12,
                 preview_width=960,
                 preview_quality=60):
        self.circle_diameter = circle_diameter
        self.band = band
        self.hue = int(hue)
        self.hue_width = int(hue_width)
        self.sat_min = int(sat_min)
        self.val_min = int(val_min)
        self.min_area_px = min_area_px
        self.min_coverage_deg = min_coverage_deg
        self.expected_range = expected_range
        self.range_tol = range_tol

        self.video_out = os.path.expanduser(video_out)
        self.csv_out = os.path.expanduser(csv_out)
        self.print_every = print_every
        self.write_queue_size = write_queue_size

        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        self.capture_fps = capture_fps
        self.frame_stride = max(1, int(frame_stride))
        self.fps = capture_fps / self.frame_stride

        self.exposure_abs = exposure_abs
        self.gain = gain

        self.preview_port = preview_port
        self.preview_fps = preview_fps
        self.preview_width = preview_width
        self.preview_quality = preview_quality
        self._preview = None

        with open(os.path.expanduser(calibration_path)) as f:
            calib = json.load(f)

        # camera calibration information
        self.mtx = np.array(calib["mtx"], dtype=np.float64)
        self.dist = np.array(calib["dist"], dtype=np.float64)
        self.focal_px = 0.5*(self.mtx[0, 0] + self.mtx[1, 1])

        # closes the gaps a thin ring breaks into
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (morph_px, morph_px))

        self.camera_index = 0
        self.cap = None
        self.writer = None
        self.csv_file = None
        self.csv_writer = None
        self.frame_idx = 0
        self._t0 = None
        # recent frame times for display and debugging
        self._frame_times = deque(maxlen=64)
        # (frame_idx, detection) for the control loop
        self._latest = (-1, None)

        self._grabber = None
        self._awriter = None
        self._stop_requested = False

    def color_mask(self, frame):
        """
        Generates a stencil from the image with the pixels of interest.
        Checks if the hue band crosses the HSV 0.
        Applies a morphological transformation to the mask in an attempt to close any gaps between ring arcs.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo_hue = (self.hue - self.hue_width) % 180
        hi_hue = (self.hue + self.hue_width) % 180

        if lo_hue <= hi_hue:
            mask = cv2.inRange(
                hsv, (lo_hue, self.sat_min, self.val_min), (hi_hue, 255, 255))
        else:
            mask = cv2.inRange(
                hsv, (0, self.sat_min, self.val_min), (hi_hue, 255, 255))
            mask |= cv2.inRange(
                hsv, (lo_hue, self.sat_min, self.val_min), (179, 255, 255))

        # closes the mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        return mask

    @staticmethod
    def _coverage_deg(pts, cx, cy, bins=36):
        """
        Checks how far the detected points wrap around a circle.
        """
        # the angle form teh fitted center
        ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)

        # checks which 10 degree wedges of the circle the arc landed in
        hit = np.zeros(bins, dtype=bool)
        hit[((ang + np.pi)/(2*np.pi)*bins).astype(int) % bins] = True

        coverage = 360*hit.sum()/bins

        return coverage

    @staticmethod
    def _fit_circle(pts):
        """
        Least-squares circle through points
        Kasa method of circle fitting.

        Basically,

        (x -  cx)^2 + (y - cy)^2 = r^2

        rearrange:

        x^2 + y^2 = 2cx x + 2cy y + r^2 - cx^2 - cy^2

        then, we use least-squares to solve for 2cx, 2cy, and r^2 - cx^2 - cy^2.
        These are the coefficients of A = [x,y,1]

        where Ax = b

        Then, cx, cy, and r are trivially obtained from the resulting coefficients.
        """

        # x and y arrays of points and converts the findContours integers to floats for numpy
        x = pts[:, 0].astype(float)
        y = pts[:, 1].astype(float)

        # sets up the algebraic solve
        A = np.column_stack([x, y, np.ones(len(x))])
        sol, *_ = np.linalg.lstsq(A, x*x + y*y, rcond=None)

        cx, cy = sol[0]/2, sol[1]/2
        r = np.sqrt(max(sol[2] + cx*cx + cy*cy, 0))

        return cx, cy, r

    @staticmethod
    def _edge_points(labels, stats, k):
        """
        Retuns the points that we are going to try and fit a circle to later.
        """
        # gets the blob's bounding box
        x, y = stats[k, cv2.CC_STAT_LEFT], stats[k, cv2.CC_STAT_TOP]
        w, h = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]

        # crops the search area to the blob bounding box for efficiency and only analyzes the points for the specific label map
        sub = (labels[y:y+h, x:x+w] == k).astype(np.uint8)

        # gives the outer and inner edge contours of the annulus and keeps every boundary pixel for precision
        contours, _ = cv2.findContours(
            sub, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # puts the points back into the full space of the frame
        pts = np.vstack([c.reshape(-1, 2) for c in contours]) + (x, y)

        return pts

    def range_from_radius(self, r):
        """
        Distance to the marker from the radius fitted to its band
        """
        rng = self.focal_px*(self.circle_diameter - self.band)/(2*r)

        return rng

    def _range_expected(self, rng):
        """
        Whether a range matches the tether length
        """
        if not self.expected_range:
            return True

        off = abs(rng - self.expected_range)

        return off <= self.range_tol*self.expected_range

    def find_blob(self, mask):
        """
        """
        # group the white pixels that are touching each other into blobs
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

        # find the best blob
        best = None

        for k in range(1, n):

            # discard every blob that does not meet a minimum px area
            area = int(stats[k, cv2.CC_STAT_AREA])
            if area < self.min_area_px:
                continue

            # get the edge poitns of the blob and try and fit a circle
            pts = self._edge_points(labels, stats, k)
            cx, cy, r = self._fit_circle(pts)
            if r <= 0:
                continue

            # gets the range from circle for debugging and validation
            rng = self.range_from_radius(r)
            if not self._range_expected(rng):
                continue

            # keeps the largest arc as the best blob
            if best is None or area > best[0]:
                best = (area, cx, cy, r, rng, pts)

        # nothing detected, wait for next frame
        if best is None:
            return None

        area, cx, cy, r, rng, pts = best

        # get the arc length
        coverage = self._coverage_deg(pts, cx, cy)

        return cx, cy, r, rng, coverage, area

    def detect(self, frame):
        """
        Finds the ring that we are trying to track
        """
        # first make a stencil mask of the frame
        mask = self.color_mask(frame)

        # find the blob that is best for measuring the payload's position
        blob = self.find_blob(mask)

        if blob is None:
            return None

        cx, cy, radius, rng, coverage, area = blob

        # check if the arc meets the minimum arc length requirement
        if coverage < self.min_coverage_deg:
            return None

        # convert pixel coords to a unit vector
        xy = cv2.undistortPoints(
            np.array([[[cx, cy]]], dtype=np.float64), self.mtx, self.dist).ravel()
        ray = np.array([xy[0], xy[1], 1])
        ray = ray/np.linalg.norm(ray)

        # the payload measurement in the camera frame
        p_C = rng*ray

        # collect detection information into a dictionary
        det = dict(u=cx, v=cy, radius=radius, coverage=coverage,
                   n_px=int(area), range_m=rng, p_C=p_C)

        return det

    def annotate(self, frame, det):
        """
        Draw the fitted circle and its center. Range and arc coverage are in
        the csv, so the picture stays clear enough to see the ring itself
        """
        c = (int(round(det["u"])), int(round(det["v"])))
        cv2.circle(frame, c, int(round(det["radius"])), (60, 255, 80), 2,
                   cv2.LINE_AA)
        cv2.drawMarker(frame, c, (60, 255, 80), cv2.MARKER_CROSS, 40, 2,
                       cv2.LINE_AA)

    def _set_camera_controls(self, camera_index):
        """
        Find exposure and gain on the driver. Must run after the stream is live,
        since setting the format resets controls on many UVC drivers.
        """
        print(apply_camera_controls(
            camera_index,
            AUTO if self.exposure_abs is None else self.exposure_abs,
            self.gain))

    def set_controls(self, exposure_abs=None, gain=None):
        """
        Change exposure or gain while recording, from the preview sliders
        """
        if exposure_abs is not None:
            self.exposure_abs = exposure_abs
        if gain is not None:
            self.gain = gain

        return apply_camera_controls(self.camera_index, exposure_abs, gain)

    def open(self, camera_index=0):
        """
        Open the camera and the output video/CSV files
        """
        # before the VideoWriter: given a path it cannot write, OpenCV falls
        # back to its image-sequence writer and warns instead of failing
        os.makedirs(os.path.dirname(self.video_out) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.csv_out) or ".", exist_ok=True)

        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc(*self.pixel_format))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.capture_fps)

        frame = None
        deadline = time.time() + 3
        while time.time() < deadline:
            try:
                ok, f = self.cap.read()
            except cv2.error:
                ok, f = False, None
            if ok and f is not None and f.size > 0:
                frame = f
                break
            time.sleep(0.05)
        if frame is None:
            self.cap.release()
            self.cap = None
            raise RuntimeError(
                "camera did not produce a valid frame during warm-up "
                "(check it isn't held by another process)")

        cc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        h, w = frame.shape[:2]
        print("negotiated: {} {}x{} @ {}".format(
            "".join(chr((cc >> 8 * i) & 0xFF) for i in range(4)),
            w, h, self.cap.get(cv2.CAP_PROP_FPS)))
        if self.frame_stride > 1:
            print(
                f"stride {self.frame_stride} -> recording at {self.fps:g} fps")
        assert (w, h) == (self.width, self.height), \
            f"requested {self.width}x{self.height}, got {w}x{h}; range wrong"

        self._set_camera_controls(camera_index)

        self.writer = cv2.VideoWriter(
            self.video_out, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (w, h))
        self._awriter = _AsyncWriter(self.writer,
                                     maxsize=self.write_queue_size)
        self._awriter.start()

        if self.preview_port is not None:
            self._preview = _MJPEGPreview(
                port=self.preview_port, max_width=self.preview_width,
                fps=self.preview_fps, quality=self.preview_quality,
                on_control=self.set_controls, on_stats=self.stats_line,
                exposure_abs=self.exposure_abs, gain=self.gain)
            self._preview.start()
            print(f"live preview: http://<co-computer-ip>:{self.preview_port}/  "
                  f"(or  ssh -L {self.preview_port}:localhost:{self.preview_port} "
                  f"user@co-computer  then http://localhost:{self.preview_port}/)")

        self.csv_file = open(self.csv_out, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["frame", "wall_time", "time_s",
                                  "u_px", "v_px", "radius_px", "coverage_deg",
                                  "n_px", "x", "y", "z", "range_m"])

        self.frame_idx = 0
        self._latest = (-1, None)
        self._t0 = time.time()

        return self

    def process_frame(self, frame):
        """
        Find the ring, annotate the frame, write the video and a CSV row
        """
        det = self.detect(frame)
        if det is not None:
            self.annotate(frame, det)

        verbose = self.print_every and (self.frame_idx % self.print_every == 0)
        now = time.time()
        t = now - self._t0

        if det is None:
            if verbose:
                print("no circle detected")
            self.csv_writer.writerow(
                [self.frame_idx, f"{now:.4f}", f"{t:.4f}"] + [NAN]*9)
        else:
            x, y, z = det["p_C"]
            if verbose:
                print(f"center=({det['u']:7.1f},{det['v']:7.1f}) px  "
                      f"r={det['radius']:5.1f} px  "
                      f"arc={det['coverage']:3.0f} deg  "
                      f"range={det['range_m']:.3f} m")
            self.csv_writer.writerow(
                [self.frame_idx, f"{now:.4f}", f"{t:.4f}",
                 f"{det['u']:.2f}", f"{det['v']:.2f}",
                 f"{det['radius']:.2f}", f"{det['coverage']:.1f}",
                 det['n_px'],
                 f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{det['range_m']:.6f}"])

        self._frame_times.append(now)
        if self._preview is not None:
            self._preview.update(frame)
        self._latest = (self.frame_idx, det)
        self._awriter.submit(frame)
        self.frame_idx += 1

        return det

    def fps_now(self):
        """
        Frames per second over the last few dozen frames, 0 before there are two
        """
        t = self._frame_times
        if len(t) < 2 or t[-1] <= t[0]:
            return 0

        rate = (len(t) - 1)/(t[-1] - t[0])

        return rate

    def stats_line(self):
        """
        One line of live status for the preview page, detection included since
        that is what the sliders are being tuned against
        """
        dropped = self._grabber.dropped if self._grabber is not None else 0
        _, det = self._latest
        seen = (f"ring {det['coverage']:.0f} deg at {det['range_m']:.2f} m"
                if det else "no ring")

        return (f"{self.fps_now():.1f} fps   frame {self.frame_idx}   "
                f"dropped {dropped}   {seen}")

    def latest_detection(self):
        """
        Most recent (frame_idx, detection) for a consumer on another thread.
        Same contract as MarkerPoseRecorder.latest_poses, and the detection is
        None when the last frame had no ring in it.
        """
        return self._latest

    def run(self, camera_index=0):
        """
        Open everything and record until Ctrl+C or camera failure
        """
        self._stop_requested = False

        def _on_sigint(signum, frame):
            self._stop_requested = True
            if self._grabber is not None:
                self._grabber.stop()

        try:
            prev_handler = signal.getsignal(signal.SIGINT)
        except (ValueError, TypeError):
            prev_handler = None

        seq = 0
        grabbed = 0
        try:
            self.open(camera_index)
            self._grabber = _FrameGrabber(self.cap)
            self._grabber.start()
            try:
                signal.signal(signal.SIGINT, _on_sigint)
            except (ValueError, RuntimeError):
                pass
            print("recording... press Ctrl+C to stop")
            while not self._stop_requested:
                frame, seq = self._grabber.read(seq)
                if frame is None:
                    break
                if grabbed % self.frame_stride == 0:
                    self.process_frame(frame)
                grabbed += 1
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.close()
            finally:
                if prev_handler is not None:
                    try:
                        signal.signal(signal.SIGINT, prev_handler)
                    except (ValueError, RuntimeError):
                        pass

    def stop(self):
        """
        Request the run() loop to exit. Safe to call from another thread.
        """
        self._stop_requested = True
        if self._grabber is not None:
            self._grabber.stop()

    def close(self):
        grabbed_ok = dropped = 0
        if self._grabber is not None:
            self._grabber.stop()
            self._grabber.join(timeout=1)
            grabbed_ok = self._grabber._seq
            dropped = self._grabber.dropped
            self._grabber = None
        if self._preview is not None:
            self._preview.stop()
            self._preview = None
        if self._awriter is not None:
            self._awriter.close()
            if self._awriter.max_depth >= self.write_queue_size - 1:
                print("warning: video write queue saturated, encode is a "
                      "bottleneck; lower resolution or raise write_queue_size")
            self._awriter = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None

        if self._t0 is not None and self.frame_idx > 0:
            elapsed = time.time() - self._t0
            achieved = self.frame_idx/elapsed if elapsed > 0 else 0
            print(f"saved {self.frame_idx} frames to {self.video_out} "
                  f"and circles to {self.csv_out}")
            print(f"achieved {achieved:.1f} fps "
                  f"(video header says {self.fps:g} fps, "
                  f"CSV time_s is the ground truth)")
            attempted = grabbed_ok + dropped
            if dropped:
                pct = 100*dropped/attempted if attempted else 0
                print(f"dropped {dropped} of {attempted} frames at the camera "
                      f"read ({pct:.0f}%), empty or corrupt buffers")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
