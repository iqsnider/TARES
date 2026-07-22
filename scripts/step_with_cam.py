# mission control imports
import comms.common as comms
from comms.control import ControlComms

# autonomy research imports
import sim.trajectory as mission
import sim.SITL_dynamics as dynamics

# logging
from logs.flight import FlightLogger

# payload camera
from payload_tracking.aruco_lib import MarkerPoseRecorder

import os
import time
import threading
from datetime import datetime


def start_camera(marker_size_m, video_out, csv_out, marker_ids=None, fps=48,
                 camera_index=0, ready_timeout=5.0, preview_port=None):
    """Launch the ArUco pose recorder on a background thread.

    marker_ids: iterable of ints to track exclusively, or None for all. When
    set, only those IDs are pose-solved, drawn, and written to the CSV --
    spurious detections of other IDs are dropped before they reach the data.

    preview_port: if set, serve a downscaled MJPEG live view on that port (view
    in a browser, ideally tunneled over SSH: ssh -L PORT:localhost:PORT ...).
    None disables it.

    The recorder writes ONLY to its own video + pose CSV. It is created with
    flight_logger=None and run with mav=None, so it never touches the flight
    logger or the mav link -- it shares no state with the control loop and
    runs fully independently. Per-frame prints are silenced (print_every=0) so
    the console stays focused on control diagnostics during the flight.

    Returns (recorder, thread). If the camera fails to come up within
    ready_timeout, prints a warning and returns anyway so the mission can
    still proceed; the caller decides whether to continue.
    """
    os.makedirs(os.path.dirname(video_out) or ".", exist_ok=True)

    recorder = MarkerPoseRecorder(
        marker_size_m=marker_size_m,
        video_out=video_out,
        csv_out=csv_out,
        fps=fps,
        marker_ids=marker_ids,       # track only these IDs (None = all)
        flight_logger=None,          # no flight-logger integration (by design)
        print_every=0,               # silent per-frame; close() still prints fps
        preview_port=preview_port)   # None = no live view

    thread = threading.Thread(
        target=recorder.run,
        kwargs={"camera_index": camera_index, "mav": None},
        daemon=True)
    thread.start()

    # wait until the camera is actually streaming (frame_idx starts advancing)
    deadline = time.time() + ready_timeout
    while recorder.frame_idx == 0 and thread.is_alive() and time.time() < deadline:
        time.sleep(0.05)

    if recorder.frame_idx == 0:
        print("WARNING: camera did not start streaming -- continuing without it")
    else:
        # recorder._t0 is the wall-clock instant (time.time()) that the pose
        # CSV's time_s column is measured from. Printing it here gives you the
        # absolute anchor to line the pose CSV up against the flight log's
        # wall_time column later, without merging the two files yet.
        print(f"camera recording started (pose t0 wall = {recorder._t0:.4f})")
    return recorder, thread


if __name__ == '__main__':
    # connection = "/dev/ttyACM0"
    # baud = 115200
    connection = "udp:127.0.0.1:14550"
    takeoff_altitude = 15
    control_freq = 50
    speed = 0.5

    # marker / camera-output config
    marker_size_m = 0.17
    track_marker_ids = [219, 220, 221]   # only these are logged; others dropped
    preview_port =  None                 # live MJPEG view; None to disable
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"camera_test_data/step_test_with_camera_{stamp}"
    video_out = f"{data_dir}/recording.avi"
    poses_out = f"{data_dir}/poses.csv"

    # add baud here if connected to real drone
    m = comms.connect(connection)

    # check for armability
    comms.wait_until_armable(m)

    # tell ardupilot not to help the external control system with the GUID_OPTION mode of 48
    comms.set_guid_options(m, 48)

    # GUIDED mode is easiest for external commands
    comms.set_mode(m, "GUIDED")

    # arm the motors if not already armed
    comms.arm(m)

    # CLEAR THE AREA
    comms.takeoff(m, takeoff_altitude)

    # intialize the logs
    logger = FlightLogger()

    # initalize control communications and prepare datastream for high rate control requests
    # rec=False: the camera is run separately on its own thread, not by ControlComms
    controlLink = ControlComms(m,
                               control_frequency=control_freq,
                               logger=logger,
                               rec=False)

    # mission reference
    startPointHoverTime = 10
    endPointHoverTime = 10

    # ENU to ENU
    ref = mission.SafeTrajectory(m, None, [10, 0, 0], speed=speed,
                                 startPointHoverTime=startPointHoverTime,
                                 endPointHoverTime=endPointHoverTime,
                                 startFromCurrentPosition=True,
                                 relativeEnd=True,
                                 logger=logger).drone_trajectory()

    # define outer-loop control law
    controller = dynamics.OuterLoopLQR()

    # start the payload camera on its own thread, scoped to the controlled flight.
    # (move this above comms.arm() if you'd rather confirm the camera is
    #  recording before committing to a takeoff.)
    recorder, cam_thread = start_camera(
        marker_size_m=marker_size_m,
        video_out=video_out,
        csv_out=poses_out,
        marker_ids=track_marker_ids,
        preview_port=preview_port,
        fps=48)

    # run autonomy
    print("running custom controller...")
    try:
        controlLink.fly_drone_trajectory(ref,
                                         controller,
                                         duration=ref.duration,
                                         yaw_lock=True,
                                         reassert=True)
    finally:
        # stop the camera first so it flushes its video + pose CSV,
        # then close the flight logger
        recorder.stop()
        cam_thread.join(timeout=5.0)
        logger.close()
