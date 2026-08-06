"""
This test is for running the closed-loop payload reference tracker and payload lqr control system with payload state estimator
"""
# mission control imports
import comms.common as comms
from comms.control import ControlComms

# autonomy research imports
import comms.trajectory as mission
import sim.dynamics as dynamics
import comms.camera as cam
import comms.estimator as est
import Prm.config as config

# logging
from logs.flight import FlightLogger


from datetime import datetime


if __name__ == '__main__':
    # connection = "/dev/ttyACM0"
    # baud = 115200
    connection = "udp:127.0.0.1:14550"
    takeoff_altitude = 15
    control_freq = 50
    speed = 1

    # marker / camera-output config
    marker_size_m = 0.17
    track_marker_ids = [config.LEFT_MARKER_ID, config.CENTER_MARKER_ID, config.RIGHT_MARKER_ID]   # only these are logged; others dropped
    preview_port = None                  # live MJPEG view; None to disable

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/pretest_08072026/payload_step_test_{stamp}"
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
    logger = FlightLogger(data_dir=data_dir)

    # start the camera first: the estimator seeds payload yaw from the first
    # detection, and a closed-loop payload run is not safe without it
    recorder, cam_thread = cam.start_camera(
        marker_size_m=config.MARKER_EDGE_LEN,
        video_out=video_out,
        csv_out=poses_out,
        marker_ids=track_marker_ids,
        preview_port=preview_port,
        capture_fps=48,
        frame_stride=1)

    # initalize control communications and prepare datastream for high rate control requests
    controlLink = ControlComms(m,
                               control_frequency=control_freq,
                               logger=logger)

    # mission reference
    startPointHoverTime = 5
    endPointHoverTime = 5

    # ENU to ENU. payload_trajectory() offsets one tether length below the
    # drone's current position, so the reference is where the PAYLOAD should be
    ref = mission.SafeTrajectory(m, None, [-20, 0, 0], speed=speed,
                                 startPointHoverTime=startPointHoverTime,
                                 endPointHoverTime=endPointHoverTime,
                                 startFromCurrentPosition=True,
                                 relativeEnd=True,
                                 logger=logger).payload_trajectory()

    # define outer-loop control law
    controller = dynamics.OuterLoopPayloadLQR()

    # payload swing estimator: attitude comes from the logger cache, which
    # ControlComms has already populated by blocking for the first state
    ekf = est.start_ekf(logger, recorder=recorder)

    # run autonomy
    print("running payload controller...")
    try:
        controlLink.fly_payload_trajectory(ref,
                                           controller,
                                           duration=ref.duration,
                                           recorder=recorder,
                                           ekf=ekf,
                                           yaw_lock=True,
                                           reassert=False)
    finally:
        # stop the camera first so it flushes its video and pose csv
        # then close the flight logger
        recorder.stop()
        cam_thread.join(timeout=5)
        logger.close()
