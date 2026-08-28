"""
Test for evaluating how the EKF responds to a blackout during hover.
"""
# mission control imports
import comms.common as comms
from comms.payload_autopilot import StickControl

# autonomy research imports
import sim.dynamics as dynamics
import Prm.config as config

# logging
from logs.flight import FlightLogger


from datetime import datetime

if __name__ == '__main__':
    connection = "/dev/ttyACM0"
    baud = 115200
    control_freq = config.CONTROL_FREQUENCY

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/test_08282026/payload_blackout_hover_test_{stamp}"
    video_out = f"{data_dir}/recording.avi"

    if config.EKF_SOURCE == "aruco":
        poses_out = f"{data_dir}/poses.csv"
    else:
        poses_out = f"{data_dir}/circles.csv"

    # establish connection
    m = comms.connect(connection, baud)

    # intialize the logs
    logger = FlightLogger(data_dir=data_dir)

    controlLink = StickControl(m,
                               control_freq,
                               logger,
                               data_dir,
                               stamp,
                               video_out,
                               poses_out)

    controlLaw = dynamics.OuterLoopPayloadLQI()
    logger.set_controller(controlLaw)

    # forces stick control and tests a camera blackout while hovering
    controlLink.force_stick_control(payload_controller=controlLaw,
                                    blackout_time=30,
                                    blackout_duration=3)
