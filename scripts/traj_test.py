import sim.config as config
import sim.drone_test_only_mission_manager as mission
import sim.SITL_dynamics as dynamics

import comms.common as comms
import comms.control as control


if __name__ == '__main__':
    # connection = "/dev/ttyACM0"
    # baud = 115200
    connection = "udp:127.0.0.1:14550"
    takeoff_altitude = 10

    # add baud here if connected to real drone
    m = comms.connect(connection)

    comms.wait_until_armable(m)
    comms.set_mode(m, "GUIDED")
    comms.arm(m)
    comms.takeoff(m, takeoff_altitude)

    control.request_fast_state(m, hz=50)

    # straight line
    controller = dynamics.PositionController()
    ref = mission.ReferenceTrajectory([0, 0, 10], [20, 0, 10], speed=0.5)
    control.fly_trajectory(
        m, ref, controller, duration=ref.total_time_to_wp, dt=1/50)

    comms.land(m)
