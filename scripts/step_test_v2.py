
# mission control imports
import comms.common as comms
import comms.control as control

# autonomy research imports
import sim.trajectory as mission
import sim.SITL_dynamics as dynamics


if __name__ == '__main__':
    # connection = "/dev/ttyACM0"
    # baud = 115200
    connection = "udp:127.0.0.1:14550"
    takeoff_altitude = 15
    control_freq = 50
    speed = 0.5

    # add baud here if connected to real drone
    m = comms.connect(connection)

    # check for armability
    comms.wait_until_armable(m)

    # tell ardupilot not to help the external control system
    comms.set_guid_options(m, 48)

    # GUIDED mode is easiest for external commands
    comms.set_mode(m, "GUIDED")

    # arm the motors if not already armed
    comms.arm(m)

    # CLEAR THE AREA
    comms.takeoff(m, takeoff_altitude)

    # prepare datastream for high rate control requests
    control.request_fast_state(m, hz=control_freq)

    # ------- initialize autonomy -------

    # 5 m test
    startPointHoverTime = 5
    endPointHoverTime = 5
    ref = mission.SafeTrajectory(m, [0, 0, 15], [5, 0, 15], speed=speed,
                                      startPointHoverTime=startPointHoverTime,
                                      endPointHoverTime=endPointHoverTime).drone_trajectory()




    # run autonomy
    print("running custom controller...")

    # define outer-loop control law
    controller = dynamics.OuterLoopLQR()

    control.fly_trajectory(
        m, ref, controller, duration=ref.duration, dt=1/control_freq, yaw_lock=True, reassert=True)
