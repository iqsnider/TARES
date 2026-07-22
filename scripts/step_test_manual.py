# mission control imports
import comms.common as comms
from comms.control import ControlComms

# autonomy research imports
import sim.trajectory as mission
import sim.SITL_dynamics as dynamics

# logging
from logs.flight import FlightLogger


if __name__ == '__main__':
    connection = "/dev/ttyACM0"
    baud = 115200
    # connection = "udp:127.0.0.1:14550"
    takeoff_altitude = 15
    control_freq = 50
    speed = 0.5

    # add baud here if connected to real drone
    m = comms.connect(connection, baud)

    # check for armability
    comms.wait_until_armable(m)

    # tell ardupilot not to help the external control system with the GUID_OPTION mode of 48
    comms.set_guid_options(m, 48)

    # GUIDED mode is easiest for external commands
    comms.set_mode(m, "GUIDED")

    # arm the motors if not already armed
    # comms.arm(m)

    # CLEAR THE AREA
    # comms.takeoff(m, takeoff_altitude)


    # intialize the logs
    logger = FlightLogger()

    # initalize control communications and prepare datastream for high rate control requests
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




    # run autonomy
    print("running custom controller...")

    # define outer-loop control law
    controller = dynamics.OuterLoopLQR()

    controlLink.fly_drone_trajectory(ref, 
                               controller, 
                               duration=ref.duration,
                               yaw_lock=True, 
                               reassert=False)
    # close the logger
    logger.close()
