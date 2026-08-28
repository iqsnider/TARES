"""
Active payload trajectory generation from transmitter sticks.

Program for controlling the payload position from transmitter sticks in GUIDED mode.

Generates a position/velocity/acceleration reference for the payload from transmitter stick inputs.
Then uses the payload swing estimate and payload control system to control the payload.
"""
# mission control imports
from comms.control import ControlComms, get_state_enu
import comms.common as comms

# autonomy research imports
import sim.dynamics as dynamics
import sim.transformations as tf
import sim.estimation.ekf as ekfm
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
        self.roll_pwm = 1500  # roll, centers 1500
        self.pitch_pwm = 1500  # pitch, centers 1500
        # throttle, rests 1102 at the bottom TODO: figure out hover throttle
        self.throttle_pwm = 1102
        # yaw, centers 1498 (not necessary, but we'll grab it for sake of completeness)
        self.yaw_pwm = 1498
        # self.mode_pwm = 1102 # mode: 1102 LOITER, 1500 GUIDED, 1897 STABILIZE (may not be necessary, because mode information is more generally grabbed from the HEARTBEAT)

        # initialize data collection
        self.stamp = stamp
        self.data_dir = data_dir
        self.video_out = video_out
        self.poses_out = poses_out

        # initialize payload tracking, markers or a color ring per EKF_SOURCE
        self.recorder, self.cam_thread = cam.start_payload_camera(
            video_out=self.video_out,
            csv_out=self.poses_out)

        # initialize payload state estimator
        self.ekf = est.start_ekf(self.logger, recorder=self.recorder)

        # initialize mode information
        self.logger.pump(self.m)
        c = self.logger.cache
        self.mode = c["echoed_mode"]

        # making time monotonic
        self.t0 = time.time()

    def run_payload_stick_control(self,
                                  payload_controller=dynamics.OuterLoopPayloadLQR()):
        """
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

        last_seq = -1
        # the loop time the filter last integrated to; None until the first
        # pass, which has no elapsed time to measure and uses the nominal one
        t_prev = None

        # initialize the payload reference
        self.logger.pump(self.m)
        c = self.logger.cache
        x = get_state_enu(c['ned'], prev=x)
        last_seq, meas = est.latest_measurement(self.recorder, self.ekf.source)
        xi, _ = est.step_ekf(self.ekf, meas, est.accel_enu(c), dt,
                             c['roll'], c['pitch'], c['yaw'])
        self.ref_position = x[0:3] + L*np.array([xi[0], xi[1], -1])
        self.ref_velocity = np.zeros(3)
        self.ref_acceleration = np.zeros(3)
        self.thr_armed = False

        # intialize time for control loop
        next_t = time.time()

        # run payload stick control when mode is set to GUIDED
        while self.mode == "GUIDED":
            t = time.time() - self.t0
            self.logger.pump(self.m)
            c = self.logger.cache

            # get current drone state p,v
            x = get_state_enu(c['ned'], prev=x)

            # fold in the camera only when the frame is new
            seq, meas = est.latest_measurement(self.recorder, self.ekf.source)
            if seq == last_seq:
                meas = None
            else:
                last_seq = seq

            # ekf dt
            dt_ekf = dt if t_prev is None else t - t_prev
            t_prev = t

            # payload swing estimate: [alpha_x alpha_y alpha_dot_x alpha_dot_y psi_p]
            xi, P = est.step_ekf(self.ekf, meas, est.accel_enu(c), dt_ekf,
                                 c['roll'], c['pitch'], c['yaw'])

            # generate the payload reference from the stick inputs
            p_ref, v_ref, a_ref = self.stick_to_payload_ref(
                dt=dt_ekf, yaw=c["yaw"])
            x_ref = dynamics.tether_equilibrium_state(p_ref, v_ref, L)

            # assemble the measured 16-state and compute the control input
            x16 = est.payload_state_16(x, xi)
            a_des = payload_controller.compute_u(x16 - x_ref)

            # send off the bitmask to the FC
            mask = self.send_accel(tf.T_ENU_from_NED() @ a_des, yaw=yaw_ref)

            # confirm the bitmask with the FC
            self.logger.note_sent(bitmask=mask)

            # log values
            self.logger.log(t, x, u=a_des,
                            yaw_ref=yaw_ref,
                            payload_p_ref=p_ref,
                            payload_v_ref=v_ref,
                            payload_alpha=(xi[0], xi[1]),
                            payload_alphadot=(xi[2], xi[3]),
                            payload_psi_p=xi[ekfm.IX_PSI_P],
                            payload_innov=self.ekf.innov,
                            payload_cov=(P[ekfm.IX_ALPHA_X, ekfm.IX_ALPHA_X],
                                         P[ekfm.IX_ALPHA_Y, ekfm.IX_ALPHA_Y],
                                         P[ekfm.IX_ALPHA_X, ekfm.IX_ALPHA_Y],
                                         P[ekfm.IX_PSI_P, ekfm.IX_PSI_P]))

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
        x = self.x0
        try:
            while True:
                self.logger.pump(self.m)
                c = self.logger.cache

                self.mode = c["echoed_mode"]

                if self.mode == payload_control_mode:
                    print(
                        f"{payload_control_mode} mode detected, switching to payload stick control...")
                    self.run_payload_stick_control(
                        payload_controller)

                # log while idle too, so time outside the control loop is
                # visible instead of collapsing into a single row
                x = get_state_enu(c['ned'], prev=x)
                self.logger.log(time.time() - self.t0, x)

                time.sleep(1/self.hz)
        finally:
            self._close_link()

    @staticmethod
    def _norm(pwm):
        """
        Stick PWM -> [-1, 1], deadzone removed and slope corrected for it.
        """
        d = pwm - config.STICK_TRIM
        if not np.isfinite(d) or abs(d) <= config.STICK_DZ:
            return 0

        d = np.sign(d)*(abs(d) - config.STICK_DZ) / \
            (config.STICK_TRAVEL - config.STICK_DZ)

        normalized_pwm = float(np.clip(d, -1, 1))

        return normalized_pwm

    def stick_to_payload_ref(self, dt, yaw=None):
        """
        Converts stick PWM signals to payload reference position/velocity/acceleration.
        ENU if no yaw, if yaw the direction is reference to whatever direction the body is facing.
        The reference is then used for the payload controller to track.
        Sets the corresponding class attributes.
        """
        self._get_stick_signals()
        sr = self._norm(self.roll_pwm)
        sp = -self._norm(self.pitch_pwm)
        st = self._norm(self.throttle_pwm)

        # no vetical control until pilot centers stick
        if not self.thr_armed and st == 0:
            self.thr_armed = True

        v_cmd = np.zeros(3)

        # referenced to drone direction
        if yaw is not None:
            v_cmd[0] = config.PAYLOAD_V_XY_MAX * \
                (sp*np.sin(yaw) + sr*np.cos(yaw))
            v_cmd[1] = config.PAYLOAD_V_XY_MAX * \
                (sp*np.cos(yaw) - sr*np.sin(yaw))

        # ENU reference
        else:
            v_cmd[0] = config.PAYLOAD_V_XY_MAX*sr
            v_cmd[1] = config.PAYLOAD_V_XY_MAX*sp

        if self.thr_armed:
            v_cmd[2] = config.PAYLOAD_V_Z_MAX*st
        else:
            v_cmd[2] = 0

        a_max = config.MAX_STICK_ACCELERATION
        self.ref_acceleration = np.clip(
            (v_cmd - self.ref_velocity)/dt, -a_max, a_max)
        self.ref_velocity += self.ref_acceleration*dt
        self.ref_position += self.ref_velocity*dt

        return self.ref_position, self.ref_velocity, self.ref_acceleration

    def _get_stick_signals(self):
        """
        Gets the stick PWM signal and sets the corresponding class attributes
        """
        rc = self.logger.cache["rc"]
        self.roll_pwm = rc[0]  # ch1
        self.pitch_pwm = rc[1]  # ch2
        self.throttle_pwm = rc[2]  # ch3
        self.yaw_pwm = rc[3]  # ch4

    def force_stick_control(self,
                            payload_controller=dynamics.OuterLoopPayloadLQR(),
                            blackout_time=None,
                            blackout_duration=None):
        """
        FOR EXPERIMENTAL FLIGHT ONLY. DO NOT USE UNLESS YOUR NAME IS IAN SNIDER.
        """
        comms.set_mode(self.m, "GUIDED", logger=self.logger)
        # compute time step
        dt = 1/self.hz
        L = config.TETHER_LEN
        yaw_ref = self.logger.cache['yaw']

        # make sure we have full control
        comms.set_guid_options(self.m, 48)

        # get initial state
        self._wait_fresh_state()
        x = self.x0

        last_seq = -1
        # the loop time the filter last integrated to; None until the first
        # pass, which has no elapsed time to measure and uses the nominal one
        t_prev = None

        # initialize the payload reference
        self.logger.pump(self.m)
        c = self.logger.cache
        x = get_state_enu(c['ned'], prev=x)
        last_seq, meas = est.latest_measurement(self.recorder, self.ekf.source)
        xi, _ = est.step_ekf(self.ekf, meas, est.accel_enu(c), dt,
                             c['roll'], c['pitch'], c['yaw'])
        self.ref_position = x[0:3] + L*np.array([xi[0], xi[1], -1])
        self.ref_velocity = np.zeros(3)
        self.ref_acceleration = np.zeros(3)
        self.thr_armed = False

        # intialize time for control loop
        next_t = time.time()

        # run payload stick control when mode is set to GUIDED
        while self.mode == "GUIDED":
            t = time.time() - self.t0
            self.logger.pump(self.m)
            c = self.logger.cache

            # get current drone state p,v
            x = get_state_enu(c['ned'], prev=x)

            if blackout_time is not None and t > blackout_time and t < (blackout_time + blackout_duration):
                seq = last_seq

            else:
                seq, meas = est.latest_measurement(
                    self.recorder, self.ekf.source)
                if seq == last_seq:
                    meas = None
                else:
                    last_seq = seq

            # ekf dt
            dt_ekf = dt if t_prev is None else t - t_prev
            t_prev = t

            # payload swing estimate: [alpha_x alpha_y alpha_dot_x alpha_dot_y psi_p]
            xi, P = est.step_ekf(self.ekf, meas, est.accel_enu(c), dt_ekf,
                                 c['roll'], c['pitch'], c['yaw'])

            # generate the payload reference from the stick inputs
            p_ref, v_ref, a_ref = self.stick_to_payload_ref(
                dt=dt_ekf, yaw=c["yaw"])
            x_ref = dynamics.tether_equilibrium_state(p_ref, v_ref, L)

            # assemble the measured 16-state and compute the control input
            x16 = est.payload_state_16(x, xi)
            a_des = payload_controller.compute_u(x16 - x_ref)

            # send off the bitmask to the FC
            mask = self.send_accel(tf.T_ENU_from_NED() @ a_des, yaw=yaw_ref)

            # confirm the bitmask with the FC
            self.logger.note_sent(bitmask=mask)

            # log values
            self.logger.log(t, x, u=a_des,
                            yaw_ref=yaw_ref,
                            payload_p_ref=p_ref,
                            payload_v_ref=v_ref,
                            payload_alpha=(xi[0], xi[1]),
                            payload_alphadot=(xi[2], xi[3]),
                            payload_psi_p=xi[ekfm.IX_PSI_P],
                            payload_innov=self.ekf.innov,
                            payload_cov=(P[ekfm.IX_ALPHA_X, ekfm.IX_ALPHA_X],
                                         P[ekfm.IX_ALPHA_Y, ekfm.IX_ALPHA_Y],
                                         P[ekfm.IX_ALPHA_X, ekfm.IX_ALPHA_Y],
                                         P[ekfm.IX_PSI_P, ekfm.IX_PSI_P]))

    def _close_link(self):
        """
        Prepares the program for ending
        """
        self.logger.pump(self.m)
        if self.logger.cache["echoed_mode"] == "GUIDED":
            comms.set_mode(self.m, "BRAKE", logger=self.logger)

        comms.set_guid_options(self.m, 0)
        self.recorder.stop()
        self.cam_thread.join(timeout=5)
        self.logger.close()
