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
                 circle_diameter_m=0.3,        # outer diameter of the ring
                 band_m=0.03,                  # width of the colored band
                 hue=110,                      # 0-179, OpenCV's half-degree hue
                 hue_width=15,
                 sat_min=90,
                 val_min=50,
                 min_area_px=150,
                 min_coverage_deg=200,         # how far the arc must wrap
                 expected_range_m=None,        # the tether length, if known
                 range_tol=0.35,               # how far the ring may be off it
                 fit_tol_px=3,                 # inlier band for the fit
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
        self.expected_range_m = expected_range_m
        self.range_tol = range_tol
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

        # auto exposure drifts the color balance, so a hue threshold set in one
        # light will not hold in another
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

        self.camera_index = 0
        self.cap = None
        self.writer = None
        self.csv_file = None
        self.csv_writer = None
        self.frame_idx = 0
        self._t0 = None
        # recent frame times, for a live rate rather than the average so far
        self._frame_times = deque(maxlen=64)
        # (frame_idx, detection) for the control loop
        self._latest = (-1, None)

        self._grabber = None
        self._awriter = None
        self._stop_requested = False

    def color_mask(self, frame):
        """
        Pixels within the hue band, cleaned up
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
        Boundaries rather than regions, because the tether and the payload cut
        the ring into arcs and three points still fix a circle.
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
        Least-squares circle through points, minimizing algebraic distance
        """
        x, y = pts[:, 0], pts[:, 1]
        A = np.column_stack([x, y, np.ones(len(x))])
        sol, *_ = np.linalg.lstsq(A, x*x + y*y, rcond=None)

        cx, cy = sol[0]/2, sol[1]/2
        r = np.sqrt(max(sol[2] + cx*cx + cy*cy, 0))

        return cx, cy, r

    @staticmethod
    def _coverage_deg(pts, cx, cy, bins=36):
        """
        How far the points wrap around the circle, in degrees. A center fit
        from a short arc is badly conditioned however many points it holds,
        so this says whether to trust it, not the inlier count.
        """
        ang = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        hit = np.zeros(bins, dtype=bool)
        hit[((ang + np.pi)/(2*np.pi)*bins).astype(int) % bins] = True

        coverage = 360*hit.sum()/bins

        return coverage

    def _inliers(self, pts, cx, cy, r):
        """
        Points on the band, given a circle through it. The tolerance spans the
        whole band on purpose: cut the ring and each piece becomes one contour
        tracing both edges and two end caps, whose fit sits mid band.
        """
        band_px = 2*r*self.band_m/self.circle_diameter_m
        tol = max(self.fit_tol_px, band_px)

        on_band = np.abs(np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) - r) < tol

        return on_band

    def fit_circle(self, arcs, pts):
        """
        Circle through the ring's arcs, seeded from the arcs themselves. Every
        contour is a connected run of boundary, so each is a candidate arc: fit
        it, score it by the points it collects, refit on the winner's inliers.
        Random three point sampling failed here, since speckle elsewhere in the
        frame outnumbered the ring.
        """
        r_min = np.sqrt(self.min_area_px/np.pi)
        r_max = 0.5*np.hypot(self.width, self.height)

        best = None
        for arc in arcs:
            cx, cy, r = self._refit(arc)
            if not r_min <= r <= r_max:
                continue
            inl = self._inliers(pts, cx, cy, r)
            if inl.sum() < 8:
                continue

            # refine before judging it. A seed arc sits on one edge of the
            # band, so its radius is not the one to test against the tether,
            # and a decoy of the wrong size would otherwise win on inlier
            # count and only then be thrown out, taking the ring with it
            cx, cy, r = self._refit(pts[inl])
            if not r_min <= r <= r_max or not self._radius_expected(r):
                continue

            inl = self._inliers(pts, cx, cy, r)
            k = int(inl.sum())
            if best is None or k > best[0]:
                best = (k, cx, cy, r, inl)

        if best is None:
            return None

        # once more on everything the winner collected, so arcs on the far
        # side of an occlusion pull their weight in the center
        _, cx, cy, r, inl = best
        cx, cy, r = self._refit(pts[inl])
        if not self._radius_expected(r):
            return None

        inliers = pts[self._inliers(pts, cx, cy, r)]

        return cx, cy, r, inliers

    def _radius_expected(self, r):
        """
        Whether a fitted radius matches the tether length, if one is known.

        The payload hangs one tether length from the camera whatever the drone
        is doing, and a swing only shortens that by cos(alpha), so the ring's
        size on the sensor is known before looking at the picture. Only the
        refitted radius is tested: a seed arc sits on one edge of the band,
        which at a short range is nowhere near the centerline the fit lands on.
        """
        if not self.expected_range_m:
            return True

        # the fit lands between the band's centerline and its outer edge, and
        # nearer the outer one, since a longer perimeter contributes more
        # boundary points. Both ends of that span have to be allowed, which
        # matters most on a wide band where they are far apart
        scale = self.focal_px/(2*self.expected_range_m)
        lo = scale*(self.circle_diameter_m - self.band_m)*(1 - self.range_tol)
        hi = scale*self.circle_diameter_m*(1 + self.range_tol)

        return lo <= r <= hi

    def detect(self, frame):
        """
        The ring's center and where it puts the payload in the camera frame.
        None when no arc of that color wraps far enough around a circle to
        place its center.
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

        # the tolerance spans the band, so the fit sits on its centerline
        range_m = self.focal_px*(self.circle_diameter_m - self.band_m)/(2*r)

        # pixel to unit ray, lens distortion undone, then out to that range
        xy = cv2.undistortPoints(np.array([[[cx, cy]]], dtype=np.float64),
                                 self.mtx, self.dist).ravel()
        ray = np.array([xy[0], xy[1], 1])
        ray /= np.linalg.norm(ray)
        p_C = range_m*ray

        det = dict(u=cx, v=cy, radius=r, coverage=coverage,
                   n_inliers=int(len(inliers)), range_m=range_m, p_C=p_C)

        return det

    def annotate(self, frame, det):
        """
        Draw the fitted circle and its center, with range and arc coverage
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
        """
        Pin exposure and gain on the driver. Must run after the stream is live,
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
            print(f"stride {self.frame_stride} -> recording at {self.fps:g} fps")
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
                                  "n_inliers", "x", "y", "z", "range_m"])

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
                 det['n_inliers'],
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
