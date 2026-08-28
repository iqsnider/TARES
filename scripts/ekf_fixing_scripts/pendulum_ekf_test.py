"""
Bench test for the payload EKF with no flight controller attached.

The camera is held by hand above a dangling payload, so the drone is pretended
stationary and level and every drone state is zero. The middle window drops the
measurements, leaving the filter to coast on its prediction, and the last one
brings them back so the correction can be seen.
"""

# autonomy research imports
import comms.estimator as est
import sim.estimation.pre_process as pp
import Prm.config as config
import comms.camera as cam

# logging
from logs.flight import FlightLogger

# math
from datetime import datetime
import time
import numpy as np


# measure, hold, measure, each for this many seconds
WINDOW_S = 20


if __name__ == '__main__':
    control_freq = config.CONTROL_FREQUENCY
    dt = 1/control_freq

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/pendulum_ekf_{stamp}"
    video_out = f"{data_dir}/recording.avi"
    poses_out = f"{data_dir}/poses.csv"

    track_marker_ids = [config.LEFT_MARKER_ID, config.CENTER_MARKER_ID,
                        config.RIGHT_MARKER_ID]   # only these are logged; others dropped
    # start camera first
    recorder, cam_thread = cam.start_camera(
        marker_size_m=config.MARKER_EDGE_LEN,
        video_out=video_out,
        csv_out=poses_out,
        marker_ids=track_marker_ids,
        preview_port=config.CAM_PREVIEW_PORT,
        capture_fps=config.CAM_FPS,
        frame_stride=config.CAM_STRIDE,
        gain=config.CAM_GAIN,
        exposure_abs=config.CAM_EXP_ABS)

    # intialize the logs
    logger = FlightLogger(data_dir=data_dir)

    # seed the cache the EKF and the logger read from with a level, stationary drone instead of pumping it
    c = logger.cache
    c["roll"], c["pitch"], c["yaw"], c["yaw_rate"] = 0, 0, 0, 0
    c["ned"] = (0,)*6
    c["imu"] = (0,)*6

    x = np.zeros(6)
    a_I = np.zeros(3)

    ekf = est.start_ekf(logger, recorder=recorder)

    t0 = time.time()
    t_prev = None
    last_seq = -1
    was_held = False
    ranges = []
    n = 0
    next_t = time.time()
    try:
        while True:
            t = time.time() - t0
            if t > 3*WINDOW_S:
                break

            hold = WINDOW_S <= t < 2*WINDOW_S
            if hold != was_held:
                print("measurements HELD" if hold else "measurements LIVE")
                was_held = hold

            # don't add stale camera frames
            seq, poses = recorder.latest_poses()
            if seq == last_seq:
                poses = None
            else:
                last_seq = seq

            # predict only while held
            if hold:
                poses = None

            # ekf dt
            dt_ekf = dt if t_prev is None else t - t_prev
            t_prev = t

            xi, P = est.step_ekf(ekf, poses, a_I, dt_ekf, 0, 0, 0)

            # raw camera range to the payload, which is the rope length once
            # it hangs still. The filter normalizes this away, so read it here
            rng = pp.range_from_poses(poses)
            if rng is not None:
                ranges.append(rng)

            n += 1
            if n % (control_freq//2) == 0:
                print(f"t {t:6.1f}  alpha {np.degrees(xi[0]):7.2f} {np.degrees(xi[1]):7.2f} deg  "
                      f"sigma {np.degrees(np.sqrt(P[0, 0])):5.2f} {
                    np.degrees(np.sqrt(P[1, 1])):5.2f} deg"
                    f"  range {rng if rng is not None else float('nan'):5.3f} m"
                    f"{'  HELD' if hold else ''}")

            # no setpoints go out here, so sent_mode carries the hold flag
            logger.note_sent(mode="HOLD" if hold else "LIVE")
            logger.log(t, x,
                       payload_alpha=(xi[0], xi[1]),
                       payload_alphadot=(xi[2], xi[3]),
                       payload_psi_p=xi[4],
                       payload_range=rng,
                       payload_innov=ekf.innov,
                       payload_cov=(P[0, 0], P[1, 1], P[0, 1], P[4, 4]))

            next_t += dt
            time.sleep(max(0, next_t - time.time()))

    finally:
        if ranges:
            r = np.array(ranges)
            print(f"\nrope length from {len(r)} frames: median {np.median(r):.3f} m, "
                  f"spread {r.min():.3f} to {r.max():.3f} m")

        recorder.stop()
        cam_thread.join(timeout=5)
        logger.close()
