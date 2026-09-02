"""
Camera roll and pitch calibration from the hanging payload.

Take off first, then run this. The drone holds still while the payload swings
about the vertical under the pivot, so the mean bearing over a hold is straight
down and whatever is left over is the mount. Holding three headings 90 degrees
apart separates the part that turns with the drone, which is the camera, from
the part that stays put, which is wind.

Prints the recommended angles and sets nothing.
"""

# mission control imports
import comms.common as comms
from comms.control import ControlComms, get_state_enu

# autonomy research imports
import comms.camera as cam
import comms.estimator as est
import sim.transformations as tf
import Prm.config as config

# logging
from logs.flight import FlightLogger


from datetime import datetime
import time

import cv2
import numpy as np
from pymavlink import mavutil


M = mavutil.mavlink

# how long each heading is watched. A 6.2 m tether is a 5 s pendulum, so 20 s
# is four swings, and it is whole swings that average out of the mean
HOLD_TIME = 20  # [s]
SETTLE_TIME = 5  # [s] after a rotation, before the samples start counting
YAW_STEP = 90  # [deg] between holds
YAW_RATE = 15  # [deg/s]
YAW_TOL = np.radians(3)


def condition_yaw(m, angle_deg, rate_deg_s):
    """
    Ardupilot's own yaw command, relative and clockwise
    """
    m.mav.command_long_send(
        m.target_system, m.target_component,
        M.MAV_CMD_CONDITION_YAW, 0,
        angle_deg, rate_deg_s, 1, 1, 0, 0, 0)


def wrap(angle):
    """
    Angle onto [-pi, pi)
    """
    wrapped = (angle + np.pi) % (2*np.pi) - np.pi

    return wrapped


def pump_for(m, logger, seconds):
    """
    Keep the link drained for a while, commanding nothing
    """
    t_end = time.time() + seconds
    while time.time() < t_end:
        logger.pump(m)
        time.sleep(0.02)


def wait_for_yaw(m, logger, target, timeout=30):
    """
    Pump until the heading settles on target [rad]
    """
    t_end = time.time() + timeout
    while time.time() < t_end and abs(wrap(logger.cache['yaw'] - target)) > YAW_TOL:
        logger.pump(m)
        time.sleep(0.02)


def observe_hold(m, logger, recorder, x, t0):
    """
    Mean payload bearing and mean attitude over one hold.

    The bearing averages (alpha_x, alpha_y), the ENU swing angles, over every
    new camera frame. Yaw averages through its sine and cosine so a hold
    sitting on the wrap does not average to the opposite heading.
    """
    S = tf.T_ENU_from_NED()
    alphas = []
    atts = []
    last_seq = -1

    t_end = time.time() + HOLD_TIME
    while time.time() < t_end:
        logger.pump(m)
        c = logger.cache

        seq, meas = est.latest_measurement(recorder, config.EKF_SOURCE)
        if seq == last_seq:
            time.sleep(0.005)
            continue
        last_seq = seq

        T_IB = S @ tf.T_IB(c['roll'], c['pitch'], c['yaw']) @ S
        angles = est.swing_angles_of(meas, T_IB, config.EKF_SOURCE)
        if angles is None:
            continue

        alphas.append(angles[0:2])
        atts.append((c['roll'], c['pitch'], c['yaw']))

        x = get_state_enu(c['ned'], prev=x)
        logger.log(time.time() - t0, x, payload_alpha=angles[0:2])

    alphas = np.array(alphas)
    atts = np.array(atts)
    mean_alpha = alphas.mean(axis=0)
    mean_att = np.array([atts[:, 0].mean(),
                         atts[:, 1].mean(),
                         np.arctan2(np.sin(atts[:, 2]).mean(),
                                    np.cos(atts[:, 2]).mean())])

    return mean_alpha, mean_att, len(alphas), x


def split_body_and_earth(means, atts):
    """
    Least squares split of the mean bearings into what turns with the drone
    and what stays with the ground.

    Each hold sees m_k = A_k c + e: c is the horizontal bearing offset in body
    axes, which is the camera mount and rotates with the heading, e is the
    horizontal lean fixed to the ground, which does not. A_k is the horizontal
    block of the body to ENU rotation that hold was seen through. Three
    headings is six equations for the four unknowns.
    """
    S = tf.T_ENU_from_NED()
    rows = []
    for phi, theta, psi in atts:
        R = (S @ tf.T_IB(phi, theta, psi) @ S)[0:2, 0:2]
        rows.append(np.hstack([R, np.eye(2)]))

    A = np.vstack(rows)
    b = np.concatenate(means)
    sol = np.linalg.lstsq(A, b, rcond=None)[0]
    resid = A @ sol - b

    return sol[0:2], sol[2:4], resid


def corrected_extrinsics(c):
    """
    The mount angles that put the observed hang direction straight down.

    The camera puts the payload at [c_x, c_y, -1] in the body frame where it
    belongs at [0, 0, -1], so the mount is off by the shortest rotation between
    those two. That rotation turns about the horizontal, so it leaves the mount
    yaw where it is, which is right: a plumb line cannot see camera yaw. Apply
    it to the current mount and read the angles back out of the Rz Rx Ry the
    extrinsics are built from.
    """
    d_obs = np.array([c[0], c[1], -1])
    d_obs = d_obs/np.linalg.norm(d_obs)
    d_des = np.array([0, 0, -1])

    axis = np.cross(d_obs, d_des)
    angle = np.arctan2(np.linalg.norm(axis), d_obs @ d_des)
    R_corr = cv2.Rodrigues(axis/np.linalg.norm(axis)*angle)[0]

    # T_BC starts from a camera already staring down, so undo that part before
    # reading the sequence back
    U = R_corr @ config.CAM_R @ np.diag([1, -1, -1])
    pitch = np.arcsin(np.clip(U[2, 1], -1, 1))
    roll = np.arctan2(-U[2, 0], U[2, 2])
    yaw = np.arctan2(-U[0, 1], U[1, 1])

    return roll, pitch, yaw


if __name__ == '__main__':
    connection = "/dev/ttyACM0"
    baud = 115200
    # connection = "udp:127.0.0.1:14550"
    control_freq = config.CONTROL_FREQUENCY

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/pretest_09012026/camera_ext_calibration_{stamp}"
    video_out = f"{data_dir}/recording.avi"

    if config.EKF_SOURCE == "aruco":
        poses_out = f"{data_dir}/poses.csv"
    else:
        poses_out = f"{data_dir}/circles.csv"

    # add baud here if connected to real drone
    m = comms.connect(connection, baud)

    # GUIDED holds position on its own, and no setpoint is sent from here: the
    # only command this script gives ardupilot is the yaw between holds
    comms.set_guid_options(m, 0)
    comms.set_mode(m, "GUIDED")

    # intialize the logs
    logger = FlightLogger(data_dir=data_dir)

    # start camera first
    recorder, cam_thread = cam.start_payload_camera(video_out=video_out,
                                                    csv_out=poses_out)

    try:
        # nothing is commanded through this, it is here for the fast state
        # streams and the first state
        controlLink = ControlComms(m,
                                   control_frequency=control_freq,
                                   logger=logger)

        x = controlLink.x0
        t0 = time.time()
        means, atts, counts = [], [], []

        for i in range(3):
            if i > 0:
                target = wrap(logger.cache['yaw'] + np.radians(YAW_STEP))
                print(f"rotating {YAW_STEP} deg...")
                condition_yaw(m, YAW_STEP, YAW_RATE)
                wait_for_yaw(m, logger, target)

                # the rotation shoves the payload, let that settle before the
                # samples start counting
                pump_for(m, logger, SETTLE_TIME)

            print(f"hold {i + 1}: watching the payload for {HOLD_TIME}s...")
            mean_alpha, mean_att, n, x = observe_hold(
                m, logger, recorder, x, t0)
            means.append(mean_alpha)
            atts.append(mean_att)
            counts.append(n)

        c, e, resid = split_body_and_earth(means, atts)
        roll, pitch, yaw = corrected_extrinsics(c)

        print("\ncamera extrinsic calibration")
        for i, (mean_alpha, mean_att, n) in enumerate(zip(means, atts, counts),
                                                      start=1):
            print(f"  hold {i}: heading {np.degrees(mean_att[2]):7.1f} deg, "
                  f"mean swing ({np.degrees(mean_alpha[0]):+6.2f}, "
                  f"{np.degrees(mean_alpha[1]):+6.2f}) deg, {n} frames")

        print(f"  turns with the drone (the camera): "
              f"({np.degrees(c[0]):+6.2f}, {np.degrees(c[1]):+6.2f}) deg")
        print(f"  stays with the ground (wind):      "
              f"({np.degrees(e[0]):+6.2f}, {np.degrees(e[1]):+6.2f}) deg")
        print(f"  largest fit residual: {
              np.degrees(np.abs(resid)).max():.2f} deg")

        print(f"\n  CAM_ROLL_DEG  {np.degrees(config.CAM_ROLL):+7.2f} "
              f"-> {np.degrees(roll):+7.2f}")
        print(f"  CAM_PITCH_DEG {np.degrees(config.CAM_PITCH):+7.2f} "
              f"-> {np.degrees(pitch):+7.2f}")
        print(f"  CAM_YAW_DEG   {np.degrees(config.CAM_YAW):+7.2f}, leave it: a "
              f"plumb line cannot see camera yaw")
        print(f"  (the correction reads yaw back as "
              f"{np.degrees(yaw):+7.2f}, which is the same angle to within the fit)")
        print(f"\n  nothing was set. Prm/airframes/{config.AIRFRAME}.json")
    finally:
        # stop the camera first so it flushes its video and pose csv
        # then close the flight logger
        recorder.stop()
        cam_thread.join(timeout=5)
        logger.close()
