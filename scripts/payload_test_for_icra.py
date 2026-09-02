"""
Payload test for ICRA data. Square flightplan
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
    connection = "/dev/ttyACM0"
    baud = 115200
    # connection = "udp:127.0.0.1:14550"
    # takeoff_altitude = 15
    control_freq = config.CONTROL_FREQUENCY
    speed = 1

    # the ramp onto speed, matched to the WPNAV_ACCEL the ardupilot baseline
    # flies so the two runs differ in the controller and nothing else
    accel = 0.5

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/test_09022026/payload_icra_test_{stamp}"
    video_out = f"{data_dir}/recording.avi"

    if config.EKF_SOURCE == "aruco":
        poses_out = f"{data_dir}/poses.csv"
    else:
        poses_out = f"{data_dir}/circles.csv"

    # add baud here if connected to real drone
    m = comms.connect(connection, baud)

    # check for armability
    # comms.wait_until_armable(m)

    # GUIDED mode is easiest for external commands
    comms.set_guid_options(m, 0)
    comms.set_mode(m, "GUIDED")

    # arm the motors if not already armed
    # comms.arm(m)
    #
    # # CLEAR THE AREA
    # comms.takeoff(m, takeoff_altitude)

    # intialize the logs
    logger = FlightLogger(data_dir=data_dir)

    # start camera first
    recorder, cam_thread = cam.start_payload_camera(
        video_out=video_out,
        csv_out=poses_out)

    controlLink = None
    try:
        # initalize control communications and prepare datastream for high rate control requests
        controlLink = ControlComms(m,
                                   control_frequency=control_freq,
                                   logger=logger)

        # mission reference
        wp_hover_time = 20

        # define outer-loop control law
        controller = dynamics.OuterLoopPayloadLQR()
        logger.set_controller(controller)

        # payload swing estimator: attitude comes from the logger cache, which
        # ControlComms has already populated by blocking for the first state
        ekf = est.start_ekf(logger, recorder=recorder)

        # ENU to ENU. One reference for the whole square, built off where the
        # payload actually starts, with every leg picking up where the last
        # one ended so the corners join
        plan = mission.SafeFlightPlan(m,
                                      wp_hover_time=wp_hover_time,
                                      speed=speed,
                                      logger=logger)
        legs = plan.payload_plan(controlLink.payload_position(ekf), accel)

        # tell ardupilot not to help the external control system with the GUID_OPTION mode of 48
        comms.set_guid_options(m, 48)

        # run autonomy: fly every leg of the flight plan in sequence
        print("running payload controller...")
        for i, ref in enumerate(legs, start=1):
            if controlLink.pilot_override:
                print("pilot override, stopping flight plan")
                break

            print(f"flying leg {i}...")
            controlLink.fly_payload_trajectory(ref,
                                               controller,
                                               duration=ref.duration,
                                               recorder=recorder,
                                               ekf=ekf,
                                               yaw_lock=True,
                                               reassert=False,
                                               anchor=False)
    finally:
        try:
            if controlLink is None or not controlLink.pilot_override:
                comms.set_mode(m, "BRAKE")
        finally:
            # flush whatever happened above: a mode change that times out
            # must not cost us the recording as well.
            # stop the camera first so it flushes its video and pose csv
            # then close the flight logger
            comms.set_guid_options(m, 0)
            recorder.stop()
            cam_thread.join(timeout=5)
            logger.close()
