"""
Active payload trajectory generation from transmitter sticks.

Program for controlling the payload position from transmitter sticks in GUIDED mode.

Generates a position/velocity/acceleration reference for the payload from transmitter stick inputs.
Then uses the payload swing estimate and payload control system to control the payload.
"""
# mission control imports
from comms.control import ControlComms
import comms.common as comms

# autonomy research imports
import sim.dynamics as dynamics
import comms.camera as cam
import comms.estimator as est
import Prm.config as config

# math imports
import numpy as np
import time


class StickControl(ControlComms):
    def __init__(self,
                 m,
                 control_frequency=50,
                 logger=None,
                 data_dir=None,
                 stamp=None,
                 video_out=None,
                 poses_out=None):
        super().__init__(m, control_frequency, logger)

        # initialize raw PWM (us) to hover values
        self.roll_pwm = 1500 # roll, centers 1500
        self.pitch_pwm = 1500 # pitch, centers 1500
        self.throttle_pwm = 1102 # throttle, rests 1102 at the bottom TODO: figure out hover throttle
        self.yaw_pwm = 1498 # yaw, centers 1498 (not necessary, but we'll grab it for sake of completeness)
        # self.mode_pwm = 1102 # mode: 1102 LOITER, 1500 GUIDED, 1897 STABILIZE (may not be necessary, because mode information is more generally grabbed from the HEARTBEAT)

        # initialize payload reference velocity and acceleration to zero
        self.ref_position = np.zeros(3) # TODO: set to current payload position
        self.ref_velocity = np.zeros(3) # initialize to 0 for hover
        self.ref_acceleration = np.zeros(3) # initialize to 0 for hover

        # initialize data collection
        self.stamp = stamp
        self.data_dir = data_dir
        self.video_out = video_out
        self.poses_out = poses_out

        # initialize marker tracking
        self.track_marker_ids = [config.LEFT_MARKER_ID, config.CENTER_MARKER_ID, config.RIGHT_MARKER_ID]
        self.recorder, self.cam_thread = cam.start_camera(marker_size_m=config.MARKER_EDGE_LEN,
                                                video_out=self.video_out,
                                                csv_out=self.poses_out,
                                                marker_ids=self.track_marker_ids,
                                                preview_port=config.CAM_PREVIEW_PORT,
                                                capture_fps=config.CAM_FPS,
                                                frame_stride=config.CAM_STRIDE,
                                                gain=config.CAM_GAIN,
                                                exposure_abs=config.CAM_EXP_ABS)

        # initialize payload state estimator
        self.ekf = est.start_ekf(self.logger, recorder=self.recorder)

        # initialize mode information
        self.logger.pump(self.m)
        c = self.logger.cache
        self.mode = c["echoed_mode"] # TODO: replace transmitter aux althold with GUIDED mode



    def run_payload_stick_control(self, payload_controller=dynamics.OuterLoopPayloadLQR()):
        """
        Should only run once in GUIDED mode.
        Initializes the payload control system and loops the stick listener
        Locks drone yaw.
        """
        # compute time step
        dt = 1/self.hz
        L = config.TETHER_LEN
        yaw_ref = self.logger.cache['yaw']

        # make sure we have full control
        comms.set_guid_options(self.m, 48)

        # get initial state
        self._wait_fresh_state()
        x = self.x0

        a_I = np.zeros(3)

        last_seq = -1
        # the loop time the filter last integrated to; None until the first
        # pass, which has no elapsed time to measure and uses the nominal one
        t_prev = None

        # intialize time for control loop
        t0 = time.time()
        next_t = t0

        # run payload stick control when mode is set to GUIDED
        while self.mode == "GUIDED":
            t = time.time() - t0
            self.logger.pump(self.m)
            c = self.logger.cache

            # get current drone state p,v
            x = self.get_state_enu(c['ned'], prev=x)

            # fold in the camera only when the frame is new
            seq, poses = self.recorder.latest_poses()
            if seq == last_seq:
                poses = None
            else:
                last_seq = seq

            # ekf dt
            dt_ekf = dt if t_prev is None else t - t_prev
            t_prev = t

            # payload swing estimate: [alpha_x alpha_y alpha_dot_x alpha_dot_y psi_p]
            xi, _ = est.step_ekf(self.ekf, poses, a_I, dt_ekf,
                                 c['roll'], c['pitch'], c['yaw'])

            # generate the payload reference from the stick inputs
            p_ref, v_ref = self.stick_to_payload_ref()
            x_ref = dynamics.tether_equilibrium_state(p_ref, v_ref, L)

            # assemble the measured 16-state and compute the control input
            x16 = est.payload_state_16(x, xi)
            u = payload_controller.compute_u(x16 - x_ref)

            # carry the commanded acceleration into the next predict step
            a_I = u

            # send off the bitmask to the FC
            mask = self.send_accel(self.enu_ned(u), yaw=yaw_ref)

            # confirm the bitmask with the FC
            self.logger.note_sent(bitmask=mask)

            # log values
            self.logger.log(t, x, u=u,
                            yaw_ref=yaw_ref,
                            payload_p_ref=p_ref,
                            payload_v_ref=v_ref,
                            payload_alpha=(xi[0], xi[1]),
                            payload_alphadot=(xi[2], xi[3]))

            # set the mode
            self.mode = c["echoed_mode"]

            # time step
            next_t += dt
            time.sleep(max(0, next_t - time.time()))


    def monitor_mode(self,
                     payload_controller=None,
                     payload_control_mode="GUIDED"):
        """
        Monitors the current mode of the flight controller.
        Will start running payload stick control if mode switches to GUIDED.
        """
        if payload_controller is None:
            payload_controller = dynamics.OuterLoopPayloadLQR()
        try:
            while True:
                self.logger.pump(self.m)
                c = self.logger.cache

                self.mode = c["echoed_mode"]

                if self.mode == payload_control_mode:
                    self.run_payload_stick_control(payload_controller)

                time.sleep(1/self.hz)
        finally:
            self._close_link()



    def stick_to_payload_ref(self):
        """
        Converts stick PWM signals to payload reference position/velocity/acceleration.
        The reference is then used for the payload controller to track.
        Sets the corresponding class attributes.
        """


    def _get_stick_signals(self):
        """
        Gets the stick PWM signal and sets the corresponding class attributes
        """
        self.logger.pump(self.m)
        c = self.logger.cache

        self.roll_pwm = c["ch1"] # roll
        self.pitch_pwm = c["ch2"] # pitch
        self.throttle_pwm = c["ch3"] # throttle
        self.yaw_pwm = c["ch4"] # yaw (not necessary, but we'll grab it for sake of completeness)


    def calibrate_stick_control(self):
        """
        Figures out the parameters needed for the correct stick to payload transfer function
        """

    def _close_link(self):
        """
        Ends the program
        """
        comms.set_mode(self.m, "BRAKE")
        comms.set_guid_options(self.m, 0)
        self.recorder.stop()
        self.cam_thread.join(timeout=5)
        self.logger.close()
