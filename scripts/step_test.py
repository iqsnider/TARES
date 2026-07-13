# mission control imports
import comms.common as comms
import comms.control as control

# autonomy research imports
import sim.drone_test_only_mission_manager as mission
import sim.SITL_dynamics as dynamics


if __name__ == '__main__':
    # connection = "/dev/ttyACM0"
    # baud = 115200
    connection = "udp:127.0.0.1:14550"
    takeoff_altitude = 10
    control_freq = 50
    speed = 0.5

    # add baud here if connected to real drone
    m = comms.connect(connection)

    # check for armability
    comms.wait_until_armable(m)

    # AUTOTUNE avoidance
    comms.set_mode(m, "LOITER")

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

    # debugging: make sure ardupilot complied and that the control rate is not being overwritten by a proxy on the same channel
    comms.check_rates(m)

    # continue hovering with ardupilot hover request
    comms.hover(m, 5) # 5 second hover command using arudpilot controller

    # ------- initialize autonomy -------

    # straight line
    controller = dynamics.OuterLoopLQR()

    # 20 m test
    ref = mission.ReferenceTrajectory([0, 0, 10], [10, 0, 10], speed=speed)

    # run autonomy
    print("running custom controller...")
    control.fly_trajectory(
        m, ref, controller, duration=ref.total_time_to_wp, dt=1/control_freq, yaw_lock=True, reassert=False)

    # ------- end autonomy -------

    # continue hovering with ardupilot hover request
    comms.hover(m, 5) # 5 second hover command using arudpilot controller

    # debugging: quickly check to make sure the streamrate didn't change again
    comms.check_rates(m)

    # No.
    # comms.land(m)
