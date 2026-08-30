"""
EKF lifecycle helpers for the swinging-payload estimator.
"""

import time

import numpy as np

import Prm.config as config
import sim.estimation.ekf as ekf_lib
import sim.estimation.pre_process as pp
import sim.transformations as tf


# RAW_IMU reports in milli-g
G_MSS = 9.80665

# full control state is 16 long; the payload swing states live at the tail
IX_ALPHA_X_16, IX_ALPHA_Y_16 = 12, 13
IX_ALPHA_DOT_X_16, IX_ALPHA_DOT_Y_16 = 14, 15


def latest_measurement(recorder, source):
    """
    Whatever the tracker last saw, in the shape its measurement builder wants
    """
    if source == ekf_lib.SOURCE_COLOR:
        return recorder.latest_detection()

    return recorder.latest_poses()


def measurement_z(meas, T_IB, source, geom=None):
    """
    The EKF measurement vector from either tracker, or None with no detection.

    A marker board gives bearing and payload yaw, a color ring gives bearing
    alone, so the vector this returns is three long or two.
    """
    geom = pp.DEFAULT_GEOMETRY if geom is None else geom
    if not meas:
        return None

    if source == ekf_lib.SOURCE_COLOR:
        return pp.measurement_from_circle(meas["p_C"], geom=geom)

    return pp.measurement_from_poses(meas, T_IB, geom=geom)


def swing_angles_of(meas, T_IB, source, geom=None):
    """
    Swing angles from either tracker's detection, or None with nothing seen.

    (alpha_x, alpha_y, psi_p) either way, though a color ring measures no
    yaw and reports 0 for it.
    """
    geom = pp.DEFAULT_GEOMETRY if geom is None else geom
    if not meas:
        return None

    if source == ekf_lib.SOURCE_COLOR:
        return pp.swing_angles_from_circle(meas["p_C"], T_IB, geom=geom)

    return pp.swing_angles_from_poses(meas, T_IB, geom=geom)


def start_ekf(logger, recorder, detect_timeout=5, source=None):
    """
    Build an EKF seeded from the current drone attitude and the first
    payload detection, from whichever tracker is running.

    Seeded at the angles that detection actually implies, not at zero: the
    stick loop takes its payload reference off this estimate the moment the
    loop starts, so a filter that begins hanging straight down puts the
    reference a whole swing away from the payload.
    """
    source = config.EKF_SOURCE if source is None else source
    c = logger.cache
    phi, theta, psi = c["roll"], c["pitch"], c["yaw"]
    S = tf.T_ENU_from_NED()
    T_IB = S @ tf.T_IB(phi, theta, psi) @ S

    deadline = time.time() + detect_timeout
    while True:
        _, meas = latest_measurement(recorder, source)
        seed = swing_angles_of(meas, T_IB, source)
        if seed is not None:
            return ekf_lib.EKF(phi, theta, psi, *seed, source=source)
        if time.time() > deadline:
            raise RuntimeError(
                f"no payload detection within {detect_timeout}s: the filter "
                f"cannot be seeded, so the payload loop must not be flown")
        time.sleep(0.05)



def accel_enu(cache):
    """
    Drone acceleration in the inertial frame, from the raw IMU.

    RAW_IMU is specific force in milli-g, body FRD. Only the horizontal
    components reach the filter and gravity leaves those alone, so it is not
    subtracted. Measured beats commanded here: the airframe does not achieve
    what the outer loop asks for, and feeding it the demand predicted the
    swing worse than feeding it nothing.
    """
    a_frd = np.array(cache["imu"][0:3], dtype=float)*G_MSS/1000
    a_ned = tf.T_IB(cache["roll"], cache["pitch"], cache["yaw"]) @ a_frd

    a_I = np.nan_to_num(tf.T_ENU_from_NED() @ a_ned)

    return a_I


def step_ekf(ekf, meas, a_I, dt, phi, theta, psi):
    """
    One predict, plus one update if a new camera frame is available.

    meas is whatever the running tracker produced: the pose dict from
    MarkerPoseRecorder, or the detection from ColorCircleRecorder.
    """
    S = tf.T_ENU_from_NED()
    ekf.T_IB = S @ tf.T_IB(phi, theta, psi) @ S
    ekf.innov = None

    xi, P = ekf.ekf_predict(ekf.xi, ekf.P, a_I, dt)

    if meas:
        z = measurement_z(meas, ekf.T_IB, ekf.source, geom=ekf.geom)
        if z is not None:
            xi, P = ekf.update_with_z(xi, P, z, ekf.T_IB)

    ekf.xi, ekf.P = xi, P

    return xi, P


def payload_state_16(x_drone, xi):
    """
    Assemble the 16 state vector OuterLoopPayloadLQR expects.
    """
    x16 = np.zeros(16)
    x16[0:6] = x_drone
    x16[IX_ALPHA_X_16] = xi[ekf_lib.IX_ALPHA_X]
    x16[IX_ALPHA_Y_16] = xi[ekf_lib.IX_ALPHA_Y]
    x16[IX_ALPHA_DOT_X_16] = xi[ekf_lib.IX_ALPHA_DOT_X]
    x16[IX_ALPHA_DOT_Y_16] = xi[ekf_lib.IX_ALPHA_DOT_Y]
    return x16
