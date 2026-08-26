"""
Track a colored ring on the payload instead of ArUco markers.

A ring only has to be segmented, not decoded, so it survives motion blur and
distance that lose a marker: the payload disc spans far more pixels than a
marker on it, and the center comes from averaging the whole boundary rather
than localizing four corners.

What it gives up is payload yaw -- a circle is rotationally symmetric -- which
the outer loop does not use. What it gives back is the center bearing and, from
the ring's apparent diameter, a range that does not depend on solving a pose
from a handful of pixels.

The camera plumbing (frame grabber, async writer, MJPEG preview) is shared with
aruco_lib rather than copied, so a fix to any of it lands in both.
"""
import csv
import json
import os
import shutil
import signal
import subprocess
import time

import cv2
import numpy as np

from payload_tracking.aruco_lib import (_AsyncWriter, _FrameGrabber,
                                        _MJPEGPreview)

NAN = float("nan")

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CALIB = os.path.join(_HERE, "camera_calibration/calibration.json")


class ColorCircleRecorder:
    def __init__(self,
                 calibration_path=_DEFAULT_CALIB,
                 circle_diameter_m=0.3,        # outer diameter of the ring
                 band_m=0.03,                  # width of the colored band
                 hue=110,                      # 0-179, OpenCV's half-degree hue
                 hue_width=15,
                 sat_min=90,
                 val_min=50,
                 min_area_px=150,
                 min_coverage_deg=200,         # how far the arc must wrap
                 fit_tol_px=3.0,               # inlier band for the fit
                 min_arc_pts=20,               # shortest contour worth fitting
                 max_fit_points=4000,
                 morph_px=3,                   # gap-closing kernel
                 video_out="recording.avi",
                 csv_out="circles.csv",
                 capture_fps=48,
                 frame_stride=1,
                 width=2304,
                 height=1536,
                 pixel_format="MJPG",
                 exposure_abs=2,               # units of 100us; 2 = 0.2 ms
                 gain=1,
                 print_every=30,
                 write_queue_size=16,
                 preview_port=None,
                 preview_fps=12,
                 preview_width=960,
                 preview_quality=60):
        self.circle_diameter_m = circle_diameter_m
        self.band_m = band_m
        self.hue = int(hue)
        self.hue_width = int(hue_width)
        self.sat_min = int(sat_min)
        self.val_min = int(val_min)
        self.min_area_px = min_area_px
        self.min_coverage_deg = min_coverage_deg
        self.fit_tol_px = fit_tol_px
        self.min_arc_pts = int(min_arc_pts)
        self.max_fit_points = int(max_fit_points)

        self.video_out = os.path.expanduser(video_out)
        self.csv_out = os.path.expanduser(csv_out)
        self.print_every = print_every            # 0 = silent per-frame
        self.write_queue_size = write_queue_size

        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        self.capture_fps = capture_fps
        self.frame_stride = max(1, int(frame_stride))
        self.fps = capture_fps / self.frame_stride

        # a short exposure matters more here than for markers: auto-exposure
        # also drifts the color balance, and a hue threshold set in one light
        # will not hold in another
        self.exposure_abs = exposure_abs
        self.gain = gain

        self.preview_port = preview_port
        self.preview_fps = preview_fps
        self.preview_width = preview_width
        self.preview_quality = preview_quality
        self._preview = None

        with open(os.path.expanduser(calibration_path)) as f:
            calib = json.load(f)
        self.mtx = np.array(calib["mtx"], dtype=np.float64)
        self.dist = np.array(calib["dist"], dtype=np.float64)
        self.focal_px = 0.5*(self.mtx[0, 0] + self.mtx[1, 1])

        # closes the gaps a thin ring breaks into
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                 (morph_px, morph_px))

        self.cap = None
        self.writer = None
        self.csv_file = None
        self.csv_writer = None
        self.frame_idx = 0
        self._t0 = None
        self._latest = (-1, None)    # (frame_idx, detection) for the control loop

        self._grabber = None
        self._awriter = None
        self._stop_requested = False

    def color_mask(self, frame):
        """
        Pixels within the hue band, cleaned up.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lo = (self.hue - self.hue_width) % 180
        hi = (self.hue + self.hue_width) % 180

        if lo <= hi:
            mask = cv2.inRange(hsv, (lo, self.sat_min, self.val_min),
                               (hi, 255, 255))
        else:
            # the band runs past 0, the way red does
            mask = cv2.inRange(hsv, (0, self.sat_min, self.val_min),
                               (hi, 255, 255))
            mask |= cv2.inRange(hsv, (lo, self.sat_min, self.val_min),
                                (179, 255, 255))

        # close only. An opening erodes a couple of pixels off every edge,
        # which erases the band entirely at range: 0.03 m is 5 px thick at
        # 8 m and 2 px at 20 m. Speckle is dropped by min_area_px instead.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        return mask

    def edge_points(self, mask):
        """
        The mask's boundaries, as a list of arcs and one stacked point array.

        Boundaries rather than regions, because the tether and the payload
        below cut the ring into arcs. Three points fix a circle, so an arc is
        as good as a whole ring -- which is what a region test cannot survive.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                       cv2.CHAIN_APPROX_NONE)
        arcs = [c.reshape(-1, 2).astype(np.float64) for c in contours
                if len(c) >= self.min_arc_pts]
        if not arcs:
            return None, None

        pts = np.vstack(arcs)
        if len(pts) > self.max_fit_points:
            pts = pts[::len(pts)//self.max_fit_points + 1]

        return arcs, pts

    @staticmethod
    def _refit(pts):
        """
        Least-squares circle through points, minimizing algebraic distance.
        """
        x, y = pts[:, 0], pts[:, 1]
        A = np.column_stack([x, y, np.ones(len(x))])
        sol, *_ = np.linalg.lstsq(A, x*x + y*y, rcond=None)
        cx, cy = sol[0]/2, sol[1]/2

        return cx, cy, np.sqrt(max(sol[2] + cx*cx + cy*cy, 0))

    @staticmethod
    def _coverage_deg(pts, cx, cy, bins=36):
        """
        How much of the circle the points actually wrap around, in degrees.

        A center fit from a short arc is badly conditioned however many points
        it holds, so this is the number that says whether to trust it -- not
        the inlier count.
        """
        ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        hit = np.zeros(bins, dtype=bool)
        hit[((ang + np.pi)/(2*np.pi)*bins).astype(int) % bins] = True

        return 360.0*hit.sum()/bins

    def fit_circle(self, arcs, pts):
        """
        Circle through the ring's arcs, seeded from the arcs themselves.

        Every contour is already a connected run of boundary, so each one is a
        candidate arc of the circle -- no random sampling needed. Fit each,
        score it by how many of all the points it collects, and refit on the
        winner's inliers, which merges arcs the occlusion split apart.

        Random three-point RANSAC was the obvious approach and it failed on
        real footage: speckle elsewhere in the room outnumbered the ring, and
        the odds of drawing three points from the same edge were 2% a go.
        """
        r_min = np.sqrt(self.min_area_px/np.pi)
        r_max = 0.5*np.hypot(self.width, self.height)

        best = None
        for arc in arcs:
            cx, cy, r = self._refit(arc)
            if not r_min <= r <= r_max:
                continue
            inl = self._inliers(pts, cx, cy, r)
            k = int(inl.sum())
            if best is None or k > best[0]:
                best = (k, inl)

        if best is None or best[0] < 8:
            return None

        # refit on everything the winning circle collected, then once more, so
        # arcs on the far side of an occlusion pull their weight in the center
        cx, cy, r = self._refit(pts[best[1]])
        inl = self._inliers(pts, cx, cy, r)
        if inl.sum() < 8:
            return None
        cx, cy, r = self._refit(pts[inl])
        if not r_min <= r <= r_max:
            return None

        return cx, cy, r, pts[inl]

    def _inliers(self, pts, cx, cy, r):
        """
        Points on the band, given a circle through it.

        The tolerance spans the whole band on purpose. Cut the ring anywhere
        and each piece becomes one contour tracing its outer edge, its inner
        edge and both end caps -- a mixture of radii whose fit sits mid-band.
        A tolerance narrower than the band would collect none of it.
        """
        band_px = 2*r*self.band_m/self.circle_diameter_m
        tol = max(self.fit_tol_px, band_px)

        return np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) < tol

    def detect(self, frame):
        """
        The ring's center and where it puts the payload in the camera frame.

        Returns None when no arc of that color wraps far enough around a circle
        to place its center. Survives the tether crossing the ring and the
        payload hiding part of it, since only the visible arc is fitted.
        """
        mask = self.color_mask(frame)
        arcs, pts = self.edge_points(mask)
        if arcs is None:
            return None

        fit = self.fit_circle(arcs, pts)
        if fit is None:
            return None
        cx, cy, r, inliers = fit

        coverage = self._coverage_deg(inliers, cx, cy)
        if coverage < self.min_coverage_deg:
            return None

        # the fit sits on the band centerline, since the tolerance spans the
        # whole band, so the centerline diameter is what the radius measures
        range_m = self.focal_px*(self.circle_diameter_m - self.band_m)/(2*r)

        # pixel -> unit ray, lens distortion undone, then out to the range the
        # apparent diameter implies
        xy = cv2.undistortPoints(np.array([[[cx, cy]]], dtype=np.float64),
                                 self.mtx, self.dist).ravel()
        ray = np.array([xy[0], xy[1], 1.0])
        ray /= np.linalg.norm(ray)
        p_C = range_m*ray

        return dict(u=cx, v=cy, radius=r, coverage=coverage,
                    n_inliers=int(len(inliers)), range_m=range_m, p_C=p_C)

    def annotate(self, frame, det):
        """
        Draw the fitted circle and its center, with range and arc coverage.
        """
        c = (int(round(det["u"])), int(round(det["v"])))
        cv2.circle(frame, c, int(round(det["radius"])), (60, 255, 80), 2,
                   cv2.LINE_AA)
        cv2.drawMarker(frame, c, (60, 255, 80), cv2.MARKER_CROSS, 40, 2,
                       cv2.LINE_AA)
        text = f"{det['range_m']:.2f} m  {det['coverage']:.0f} deg"
        at = (c[0] + 24, c[1] - 16)
        cv2.putText(frame, text, at, cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(frame, text, at, cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (60, 255, 80), 2, cv2.LINE_AA)

    def _set_camera_controls(self, camera_index):
        """Pin exposure and gain via v4l2-ctl.

        Must run after the stream is live: setting the format resets controls
        on many UVC drivers. On a mac there is no v4l2, so set them with
        uvc-util instead and the camera keeps whatever it is already on.
        """
        if shutil.which("v4l2-ctl") is None:
            print("v4l2-ctl not found -- exposure and gain left as they are")
            return

        dev = f"/dev/video{camera_index}"
        if self.exposure_abs is not None:
            subprocess.run(["v4l2-ctl", "-d", dev, "-c", "auto_exposure=1"])
            subprocess.run(["v4l2-ctl", "-d", dev,
                            "-c", f"exposure_time_absolute={self.exposure_abs}"])
        else:
            subprocess.run(["v4l2-ctl", "-d", dev, "-c", "auto_exposure=0"])
        subprocess.run(["v4l2-ctl", "-d", dev, "-c", f"gain={self.gain}"])
        readback = subprocess.run(
            ["v4l2-ctl", "-d", dev, "-C",
             "auto_exposure,exposure_time_absolute,gain"],
            capture_output=True, text=True)
        print(readback.stdout.strip())

    def open(self, camera_index=0):
        """Open the camera and the output video/CSV files."""
        # before the VideoWriter: given a path it cannot write, OpenCV falls
        # back to its image-sequence writer and warns instead of failing
        os.makedirs(os.path.dirname(self.video_out) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(self.csv_out) or ".", exist_ok=True)

        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc(*self.pixel_format))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.capture_fps)

        frame = None
        deadline = time.time() + 3.0
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
            print(f"stride {self.frame_stride} -> recording at {self.fps:g} fps")
        assert (w, h) == (self.width, self.height), \
            f"requested {self.width}x{self.height}, got {w}x{h}; range will be wrong"

        self._set_camera_controls(camera_index)

        self.writer = cv2.VideoWriter(
            self.video_out, cv2.VideoWriter_fourcc(*"MJPG"), self.fps, (w, h))
        self._awriter = _AsyncWriter(self.writer, maxsize=self.write_queue_size)
        self._awriter.start()

        if self.preview_port is not None:
            self._preview = _MJPEGPreview(
                port=self.preview_port, max_width=self.preview_width,
                fps=self.preview_fps, quality=self.preview_quality)
            self._preview.start()
            print(f"live preview: http://<co-computer-ip>:{self.preview_port}/  "
                  f"(or  ssh -L {self.preview_port}:localhost:{self.preview_port} "
                  f"user@co-computer  then http://localhost:{self.preview_port}/)")

        self.csv_file = open(self.csv_out, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["frame", "wall_time", "time_s",
                                  "u_px", "v_px", "radius_px", "coverage_deg",
                                  "n_inliers", "x", "y", "z", "range_m"])

        self.frame_idx = 0
        self._latest = (-1, None)
        self._t0 = time.time()
        return self

    def process_frame(self, frame):
        """
        Find the ring, annotate the frame, write the video and a CSV row.
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
                 det['n_inliers'],
                 f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{det['range_m']:.6f}"])

        if self._preview is not None:
            self._preview.update(frame)
        self._latest = (self.frame_idx, det)
        self._awriter.submit(frame)
        self.frame_idx += 1

        return det

    def latest_detection(self):
        """
        Most recent (frame_idx, detection) for a consumer on another thread.

        Same contract as MarkerPoseRecorder.latest_poses: a single atomic tuple
        rebind, and the frame_idx lets a faster consumer tell a new detection
        from one it has already used. The detection is None when the last frame
        had no ring in it.
        """
        return self._latest

    def run(self, camera_index=0):
        """Open everything and record until Ctrl+C or camera failure."""
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
                print("warning: video write queue saturated -- encode is a "
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
                  f"(video header says {self.fps:g} fps -- "
                  f"CSV time_s is the ground truth)")
            attempted = grabbed_ok + dropped
            if dropped:
                pct = 100*dropped/attempted if attempted else 0
                print(f"dropped {dropped} of {attempted} frames at the camera "
                      f"read ({pct:.0f}%) -- empty/corrupt buffers, not a code error")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
