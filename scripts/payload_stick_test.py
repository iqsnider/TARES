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
    # connection = "udp:127.0.0.1:14550"
    # takeoff_altitude = 15
    control_freq = config.CONTROL_FREQUENCY

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/test_09022026/stick_test_{stamp}"
    video_out = f"{data_dir}/recording.avi"

    if config.EKF_SOURCE == "aruco":
        poses_out = f"{data_dir}/poses.csv"
    else:
        poses_out = f"{data_dir}/circles.csv"

    # add baud here if connected to real drone
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

    print("monitoring for GUIDED mode...")
    # monitors the mode and swaps to payload stick control when in GUIDED
    controlLink.monitor_mode(payload_controller=controlLaw)
