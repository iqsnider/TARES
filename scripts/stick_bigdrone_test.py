"""
bigdrone
This test is for calibrating the stick signals from the payload response. Basically, just want to set the stick to payload transfer function such that the payload doesn't move around too fast or too slow.

The idea here is that the pilot should focus on the tethered payload position. The LQR then helps the pilot keep the payload where they want it. Moving the stick on the transmitter basically just moves the reference point that the LQR is meant to track.
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
    # connection = "udp:127.0.0.1:14550"
    # takeoff_altitude = 15
    control_freq = config.CONTROL_FREQUENCY

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/test_08192026/stick_bigdrone_test_{stamp}"
    video_out = f"{data_dir}/recording.avi"
    poses_out = f"{data_dir}/poses.csv"

    # add baud here if connected to real drone
    m = comms.connect(connection, baud)

    # check for armability
    # comms.wait_until_armable(m)

    # intialize the logs
    logger = FlightLogger(data_dir=data_dir)

    controlLink = StickControl(m,
                               control_freq,
                               logger,
                               data_dir,
                               stamp,
                               video_out,
                               poses_out)

    controlLaw = dynamics.OuterLoopPayloadLQR()

    print("monitoring for GUIDED mode...")
    # monitors the mode and swaps to payload stick control when in GUIDED
    comms.set_mode(m, "GUIDED", logger=logger)
    controlLink.monitor_mode(payload_controller=controlLaw)
