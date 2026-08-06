import cv2
import csv
import json
import time
import queue
import signal
import threading
import subprocess
import http.server
import numpy as np

import os

NAN = float("nan")

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CALIB = os.path.join(_HERE, "camera_calibration/calibration.json")


class _FrameGrabber(threading.Thread):
    """Continuously pull frames off the camera in the background.

    Keeps only the most recent frame (drop-freshest). For tracking you want
    the latest frame, not a backlog, so if the main loop falls behind, stale
    frames are discarded rather than queued. read() blocks until a frame newer
    than the caller's last sequence number is available, or the camera stops.
    """

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self._cond = threading.Condition()
        self._frame = None
        self._seq = 0
        self._stopped = False

    def run(self):
        fails = 0
        try:
            while True:
                # MJPG devices occasionally hand back an empty buffer, which
                # makes OpenCV's JPEG decode raise instead of returning False.
                # Treat that as a dropped frame, not a fatal error.
                try:
                    ok, frame = self.cap.read()
                except cv2.error:
                    ok, frame = False, None
                if not ok or frame is None or frame.size == 0:
                    fails += 1
                    if fails > 50:                # sustained failure = real stop
                        break
                    time.sleep(0.005)
                    continue
                fails = 0
                with self._cond:
                    if self._stopped:
                        break
                    self._frame = frame
                    self._seq += 1
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._stopped = True
                self._cond.notify_all()

    def read(self, last_seq):
        """Return (frame, seq) once a frame newer than last_seq exists.

        Returns (None, last_seq) if the camera has stopped and nothing newer
        is available.
        """
        with self._cond:
            while self._seq == last_seq and not self._stopped:
                self._cond.wait()
            if self._seq == last_seq:            # stopped, no new frame
                return None, last_seq
            return self._frame, self._seq

    def stop(self):
        with self._cond:
            self._stopped = True
            self._cond.notify_all()


class _AsyncWriter(threading.Thread):
    """Encode + write video frames on a background thread.

    Blocking put with a bounded queue applies backpressure so the recorded
    video stays 1:1 with the CSV rows (no silent frame drops) while still
    letting MJPG encode overlap detection. If the writer itself errors, it
    switches to draining-and-dropping so the main loop can never deadlock.
    """

    def __init__(self, writer, maxsize=16):
        super().__init__(daemon=True)
        self.writer = writer
        self.q = queue.Queue(maxsize=maxsize)
        self._sentinel = object()             # unique stop token (NOT _stop:
        self.failed = False                   # that name is Thread._stop())
        self.max_depth = 0

    def run(self):
        while True:
            item = self.q.get()
            if item is self._sentinel:
                break
            try:
                self.writer.write(item)
            except Exception as e:                # noqa: BLE001
                if not self.failed:
                    print(f"video writer failed, dropping frames: {e}")
                self.failed = True

    def submit(self, frame):
        if self.failed:
            return
        self.max_depth = max(self.max_depth, self.q.qsize())
        self.q.put(frame)                         # blocking = backpressure

    def close(self):
        self.q.put(self._sentinel)
        self.join()


class _MJPEGPreview:
    """Serve the latest annotated frame as an MJPEG-over-HTTP stream.
    ssh -L <port>:localhost:<port> user@co-computer
    open http://localhost:<port>/ on the laptop.
    """

    def __init__(self, port=8080, host="0.0.0.0",
                 max_width=960, fps=12, quality=60):
        self.port = int(port)
        self.host = host
        self.max_width = int(max_width)
        self.interval = 1.0 / fps if fps and fps > 0 else 0.0
        self.quality = int(quality)
        self._raw_lock = threading.Lock()
        self._raw = None
        self._jpeg_lock = threading.Lock()
        self._jpeg = None
        self._stopped = False
        self._encoder = threading.Thread(target=self._encode_loop, daemon=True)
        self._httpd = None
        self._http_thread = None

    def start(self):
        handler = self._make_handler()
        self._httpd = http.server.ThreadingHTTPServer((self.host, self.port), handler)
        self._httpd.daemon_threads = True
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True)
        self._encoder.start()
        self._http_thread.start()
        return self

    def update(self, frame):
        # hot-loop safe: just stash the latest reference; encode happens elsewhere
        with self._raw_lock:
            self._raw = frame

    def _encode_loop(self):
        while not self._stopped:
            t0 = time.time()
            with self._raw_lock:
                frame = self._raw
            if frame is not None:
                img = frame
                h, w = img.shape[:2]
                if w > self.max_width:
                    s = self.max_width / float(w)
                    img = cv2.resize(img, (self.max_width, int(round(h * s))))
                ok, buf = cv2.imencode(
                    ".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                if ok:
                    with self._jpeg_lock:
                        self._jpeg = buf.tobytes()
            rem = self.interval - (time.time() - t0)
            if rem > 0:
                time.sleep(rem)

    def _make_handler(self):
        preview = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # silence per-request console spam

            def do_GET(self):
                if self.path not in ("/", "/stream", "/stream.mjpg"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache, private")
                self.send_header("Pragma", "no-cache")
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
                self.end_headers()
                try:
                    while not preview._stopped:
                        with preview._jpeg_lock:
                            jpg = preview._jpeg
                        if jpg is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--FRAME\r\n")
                        self.wfile.write(b"Content-Type: image/jpeg\r\n")
                        self.wfile.write(
                            ("Content-Length: %d\r\n\r\n" % len(jpg)).encode())
                        self.wfile.write(jpg)
                        self.wfile.write(b"\r\n")
                        if preview.interval > 0:
                            time.sleep(preview.interval)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # client closed the tab -- not an error

        return Handler

    def stop(self):
        self._stopped = True
        if self._httpd is not None:
            try:
                self._httpd.shutdown()
                self._httpd.server_close()
            except Exception:
                pass
        if self._http_thread is not None:
            self._http_thread.join(timeout=1.0)
        if self._encoder.is_alive():
            self._encoder.join(timeout=1.0)


class MarkerPoseRecorder:
    def __init__(self,
                 calibration_path=_DEFAULT_CALIB,
                 marker_size_m=0.17,
                 video_out="recording.avi",
                 csv_out="poses.csv",
                 capture_fps=48,
                 frame_stride=1,
                 width=2304,
                 height=1536,
                 pixel_format="MJPG",
                 exposure_abs=5,               # units of 100us; 5 = 0.5 ms
                 gain=40,
                 aruco_dict=cv2.aruco.DICT_4X4_250,
                 marker_ids=None,
                 flight_logger=None,
                 print_every=30,
                 write_queue_size=16,
                 preview_port=None,
                 preview_fps=12,
                 preview_width=960,
                 preview_quality=60):
        # logging
        self.flight_logger = flight_logger
        self.mav = None

        self.marker_size_m = marker_size_m
        # markers to track; None -> accept every detected id, otherwise only these
        self.marker_ids = None if marker_ids is None else {int(i) for i in marker_ids}
        self.video_out = os.path.expanduser(video_out)
        self.csv_out = os.path.expanduser(csv_out)
        self.print_every = print_every            # 0 = silent per-frame
        self.write_queue_size = write_queue_size

        # capture format. These must be a (pixel_format, size, rate) triple the
        # driver actually lists -- check with:
        #   v4l2-ctl -d /dev/video0 --list-formats-ext
        # V4L2 snaps unsupported values silently rather than erroring, so never
        # derive these from the calibration intrinsics.
        #
        # MJPG at 2304x1536 offers 48 fps and nothing else. Lower rates come
        # from frame_stride, which keeps only every Nth frame -- so the usable
        # set is 48, 24, 16, 12. There is no 30.
        self.width = width
        self.height = height
        self.pixel_format = pixel_format
        self.capture_fps = capture_fps
        self.frame_stride = max(1, int(frame_stride))
        self.fps = capture_fps / self.frame_stride    # effective recorded rate

        # exposure is pinned manually: auto-exposure stretches integration time
        # to ~31 ms, which smears the marker under airframe vibration. Short
        # exposure + high gain trades blur for noise, and ArUco tolerates noise
        # far better than blur.
        self.exposure_abs = exposure_abs
        self.gain = gain

        # live preview (MJPEG-over-HTTP); preview_port=None disables it
        self.preview_port = preview_port
        self.preview_fps = preview_fps
        self.preview_width = preview_width
        self.preview_quality = preview_quality
        self._preview = None

        # load camera parameters
        with open(os.path.expanduser(calibration_path)) as f:
            calib = json.load(f)
        self.mtx = np.array(calib["mtx"], dtype=np.float64)
        self.dist = np.array(calib["dist"], dtype=np.float64)

        # aruco detector. Subpixel corner refinement is off by default and is
        # worth roughly 5x on pose precision at long range. The perimeter floor
        # is lowered so small (distant) markers aren't silently rejected.
        dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        params.minMarkerPerimeterRate = 0.01
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)

        # marker object points (centered square)
        s = marker_size_m / 2
        self.obj_points = np.array([[-s, s, 0],
                                    [s, s, 0],
                                    [s, -s, 0],
                                    [-s, -s, 0]], dtype=np.float32)

        # handles created when recording starts
        self.cap = None
        self.writer = None
        self.csv_file = None
        self.csv_writer = None
        self.frame_idx = 0
        self._t0 = None
        self._latest = (-1, {})      # (frame_idx, poses) for the control loop

        # threading handles
        self._grabber = None
        self._awriter = None
        self._stop_requested = False

    def get_marker_poses(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        poses = {}
        kept_corners = []
        kept_ids = []
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                mid = int(marker_id)
                # drop anything not on the whitelist (marker_ids=None -> keep all)
                if self.marker_ids is not None and mid not in self.marker_ids:
                    continue
                kept_corners.append(marker_corners)
                kept_ids.append(mid)
                ok, rvec, tvec = cv2.solvePnP(
                    self.obj_points, marker_corners[0], self.mtx, self.dist,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if ok:
                    poses[mid] = (rvec, tvec)
        # return only the kept markers so drawing, centers, and CSV see the
        # tracked set and nothing else
        if kept_ids:
            return kept_corners, np.array(kept_ids, dtype=np.int32).reshape(-1, 1), poses
        return (), None, poses

    def _set_camera_controls(self, camera_index):
        """Pin exposure and gain via v4l2-ctl.

        OpenCV's property mapping on this device is unreliable (it accepts
        out-of-range values silently), so talk to the driver directly. Must run
        after the stream is live: setting the format triggers VIDIOC_S_FMT,
        which resets controls on many UVC drivers. Two calls because
        exposure_time_absolute stays inactive until auto_exposure is switched
        to manual.
        """
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
        self.cap = cv2.VideoCapture(camera_index)
        # FOURCC first, then size, then rate -- the driver resolves them in
        # that order and a later call can invalidate an earlier one.
        self.cap.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc(*self.pixel_format))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.capture_fps)

        # Warm-up: an MJPG V4L2 device often isn't streaming valid frames for
        # the first tens of ms after set(), so the first read can come back
        # empty and OpenCV raises (-215 !buf.empty in imdecode). Retry briefly
        # instead of treating one empty buffer as fatal.
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

        # Report what the driver actually negotiated. V4L2 falls back silently
        # rather than erroring, so this is the only confirmation the requested
        # format/size/rate was honoured.
        cc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        h, w = frame.shape[:2]
        print("negotiated: {} {}x{} @ {}".format(
            "".join(chr((cc >> 8 * i) & 0xFF) for i in range(4)),
            w, h, self.cap.get(cv2.CAP_PROP_FPS)))
        if self.frame_stride > 1:
            print(f"stride {self.frame_stride} -> recording at {self.fps:g} fps")
        assert (w, h) == (self.width, self.height), \
            f"requested {self.width}x{self.height}, got {w}x{h}; pose will be wrong"

        self._set_camera_controls(camera_index)

        self.writer = cv2.VideoWriter(
            self.video_out, cv2.VideoWriter_fourcc(*"MJPG"),
            self.fps, (w, h))
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
        self.csv_writer.writerow(["frame", "wall_time", "time_s", "marker_id",
                                  "rx", "ry", "rz", "x", "y", "z", "range_m",
                                  "u_px", "v_px"])

        self.frame_idx = 0
        self._latest = (-1, {})      # drop detections from any prior session
        self._t0 = time.time()
        return self

    def process_frame(self, frame):
        """Detect markers, annotate the frame, write video + CSV rows.
        """
        # if logging drone data (kept on the main thread for timestamp alignment)
        if self.flight_logger is not None and self.mav is not None:
            self.flight_logger.pump(self.mav)
            n, e, d, vn, ve, vd = self.flight_logger.cache["ned"]
            x = [e, n, -d, ve, vn, -vd]                 # ENU
            nan3 = [float("nan")] * 3
            t_flight = time.time() - self._t0
            self.flight_logger.log(t_flight, x, nan3, nan3, nan3)

        corners, ids, poses = self.get_marker_poses(frame)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        # map marker_id -> pixel center
        centers = {}
        if ids is not None:
            for marker_corners, marker_id in zip(corners, ids.flatten()):
                c = marker_corners[0].mean(axis=0)   # mean of 4 corners -> (u, v)
                centers[int(marker_id)] = c

        verbose = self.print_every and (self.frame_idx % self.print_every == 0)
        # one clock read: wall_time joins against the flight log's wall_time
        # column, time_s stays the elapsed value measured from self._t0
        now = time.time()
        t = now - self._t0
        if poses:
            for marker_id, (rvec, tvec) in poses.items():
                cv2.drawFrameAxes(frame, self.mtx, self.dist, rvec,
                                  tvec, self.marker_size_m * 0.5)
                rx, ry, rz = rvec.flatten()
                x, y, z = tvec.flatten()
                dist_m = float(np.linalg.norm(tvec))
                if verbose:
                    print(f"id {marker_id}: x={x:+.3f} y={y:+.3f} z={z:+.3f} m  "
                          f"range={dist_m:.3f} m")
                u, v = centers[marker_id]
                self.csv_writer.writerow([self.frame_idx, f"{now:.4f}", f"{t:.4f}", marker_id,
                                          f"{rx:.6f}", f"{ry:.6f}", f"{rz:.6f}",
                                          f"{x:.6f}", f"{y:.6f}", f"{z:.6f}", f"{dist_m:.6f}",
                                          f"{u:.2f}", f"{v:.2f}"])
        else:
            nan = float("nan")
            if verbose:
                print("no marker detected")
            self.csv_writer.writerow(
                [self.frame_idx, f"{now:.4f}", f"{t:.4f}", nan,
                 nan, nan, nan, nan, nan, nan, nan, nan, nan])

        if self._preview is not None:
            self._preview.update(frame)      # before submit: stays live under writer backpressure
        # publish before submit(): submit() blocks under writer backpressure,
        # and the control loop should see the detection as soon as it exists
        self._latest = (self.frame_idx, poses)
        self._awriter.submit(frame)
        self.frame_idx += 1
        return poses

    def latest_poses(self):
        """Most recent (frame_idx, poses) for a consumer on another thread.

        No lock: this is a single tuple rebind, which is atomic under the GIL,
        and readers never mutate. The frame_idx lets a consumer running faster
        than the camera tell a new detection from one it has already used.
        """
        return self._latest

    def run(self, camera_index=0, mav=None):
        """Open everything and record until Ctrl+C or camera failure."""
        self.mav = mav
        self._stop_requested = False

        def _on_sigint(signum, frame):
            # Flip a flag and wake the grabber so the main loop exits cleanly
            # and close() flushes exactly once -- even on repeated Ctrl+C.
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
                pass                             # not on main thread -> KeyboardInterrupt path
            print("recording... press Ctrl+C to stop")
            while not self._stop_requested:
                frame, seq = self._grabber.read(seq)
                if frame is None:                # camera stopped
                    break
                # keep every Nth frame; the camera has no slower native rate
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

        Sets the stop flag and wakes the grabber so a blocked read() returns;
        run()'s finally then flushes video/CSV via close(). Idempotent, and a
        no-op if called before run() has started the grabber.
        """
        self._stop_requested = True
        if self._grabber is not None:
            self._grabber.stop()

    def close(self):
        # stop pulling new frames first, then drain what's already queued
        if self._grabber is not None:
            self._grabber.stop()
            self._grabber.join(timeout=1)
            self._grabber = None
        if self._preview is not None:
            self._preview.stop()
            self._preview = None
        if self._awriter is not None:
            self._awriter.close()               # flushes queued frames
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
            achieved = self.frame_idx / elapsed if elapsed > 0 else 0
            print(f"saved {self.frame_idx} frames to {self.video_out} "
                  f"and poses to {self.csv_out}")
            print(f"achieved {achieved:.1f} fps "
                  f"(video header says {self.fps:g} fps -- "
                  f"playback speed is only correct if these match; "
                  f"CSV time_s is the ground truth)")

    # context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


if __name__ == '__main__':
    recorder = MarkerPoseRecorder(
        marker_size_m=0.17,
        video_out="recording.avi",
        csv_out="poses.csv",
        capture_fps=48,
        frame_stride=1)          # 2 -> 24 fps, 3 -> 16 fps, 4 -> 12 fps
    recorder.run()
