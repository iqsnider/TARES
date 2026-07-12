import csv
import os
import time
from datetime import datetime
import numpy as np

from pymavlink import mavutil


# TODO later when ArUco system is working add marker detected boolean, marker px coords, estimated range/pose, kalman filter state/covariance/innovation
# loop_dt is the total time it takes to run the control loop (effectively the time between transmitting setpoints)
COLUMNS = [
    "wall_time", "cur_time", "cur_loop_dt", "cur_ctrl_freq",
    "loop_count", "fc_time_boot_ms", "hb_age_s",
    "sent_mode", "echoed_mode", "armed",
    "drone_px_ref", "drone_py_ref", "drone_pz_ref",
    "drone_vx_ref", "drone_vy_ref", "drone_vz_ref",
    "payload_px_ref", "payload_py_ref", "payload_pz_ref",
    "payload_vx_ref", "payload_vy_ref", "payload_vz_ref",
    "payload_alpha_x", "payload_alpha_y",
    "payload_alphadot_x", "payload_alphadot_y",
    "ux", "uy", "uz", "yaw_ref", "yaw_rate_ref",
    "drone_px_meas", "drone_py_meas", "drone_pz_meas",
    "drone_vx_meas", "drone_vy_meas", "drone_vz_meas",
    "drone_ax_meas", "drone_ay_meas", "drone_az_meas",
    "drone_gyrox_meas", "drone_gyroy_meas", "drone_gyroz_meas",
    "drone_roll", "drone_pitch", "drone_yaw", "drone_yaw_rate",
    "sent_setpoint_bitmask", "echoed_setpoint_bitmask",
    "setpoint_bitmask_callback_dt",
    "batt_voltage", "batt_current", "batt_rem_percent",
    "GPS_fix_type", "sat_count", "HDOP",
    "ekf_flags", "ekf_pos_horiz_var", "ekf_pos_vert_var", "ekf_vel_var",
    "ch1", "ch2", "ch3", "ch4", "ch5", "ch6", "ch7", "ch8"]

NAN = float("nan")


class FlightLogger:
    def __init__(self, data_dir="data"):

        # make and name the directory
        os.makedirs(data_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # save file with date and time

        # make and name the path to the data csv file
        self.path = os.path.join(data_dir, f"flight_{stamp}.csv")

        # open and make file writer
        self.f = open(self.path, "w", newline="")
        self.w = csv.writer(self.f)

        # initialize and write column headers (if tracking payload ref, drone ref with be `None`)
        self.w.writerow(COLUMNS)

        # AUTOTUNE investigation
        self.event_path = os.path.join(data_dir, f"flight_{stamp}_events.csv")
        self.ef = open(self.event_path, "w", newline="")
        self.ew = csv.writer(self.ef)
        self.ew.writerow(["wall_time", "fc_time_boot_ms", "severity", "text"])

        self._n = 0
        self._last_m = None  # monotonic timestamp of previous log() call

        # initialize cache
        self.cache = {
            # HEARTBEAT
            "echoed_mode": "?",
            "armed": False,
            "last_hb_wall": None,
            # ATTITUDE (radians, FC frame)
            "roll": NAN, "pitch": NAN, "yaw": NAN, "yaw_rate": NAN,
            # LOCAL_POSITION_NED (NED) (position, velocity)
            "ned": (NAN,)*6,
            "fc_time_boot_ms": NAN,
            # setpoint round-trip (sent side filled by note_sent(), echoed by pump())
            "sent_mode": "?",
            "sent_bitmask": None,
            "sent_wall": None,
            "echoed_bitmask": None,
            "bitmask_callback_dt": NAN,
            # SCALED_IMU
            "imu": (NAN,)*6,
            # SYS_STATUS
            "batt_voltage": NAN, "batt_current": NAN, "batt_rem_percent": NAN,
            # GPS_RAW_INT
            "gps_fix_type": NAN, "sat_count": NAN, "hdop": NAN,
            # EKF_STATUS_REPORT
            "ekf_flags": NAN, "ekf_pos_horiz_var": NAN,
            "ekf_pos_vert_var": NAN, "ekf_vel_var": NAN,
            # RC_CHANNELS ch1-8 of the spektrum transmitter
            "rc": [NAN]*8,
            # COMMAND_ACK for debugging
            "last_cmd": None, "last_cmd_result": None}

    def note_sent(self, bitmask=None, mode=None):
        """
        cache the sent bitmask and mode and note the time
        """
        if bitmask is not None:
            self.cache["sent_bitmask"] = bitmask
            # start of echo round-trip timer
            self.cache["sent_wall"] = time.time()
        if mode is not None:
            self.cache["sent_mode"] = mode

    def pump(self, m):
        """
        Drain the link

        Call once per loop
        """
        c = self.cache  # initialize columns from cache
        while True:
            msg = m.recv_match(blocking=False)
            if msg is None:
                break
            mtype = msg.get_type()
            if mtype == "BAD_DATA":
                continue

            # HEARTBEAT check
            if mtype == "HEARTBEAT":
                if msg.get_srcComponent() == mavutil.mavlink.MAV_COMP_ID_AUTOPILOT1:
                    # decoded string"
                    c["echoed_mode"] = m.flightmode
                    c["armed"] = m.motors_armed()
                    c["last_hb_wall"] = time.time()

            # orientation check
            elif mtype == "ATTITUDE":
                c["roll"], c["pitch"], c["yaw"] = msg.roll, msg.pitch, msg.yaw
                c["yaw_rate"] = msg.yawspeed

            # position/velocity check
            elif mtype == "LOCAL_POSITION_NED":
                c["fc_time_boot_ms"] = msg.time_boot_ms
                c["ned"] = (msg.x, msg.y, msg.z, msg.vx, msg.vy, msg.vz)

            # scaled imu check
            elif mtype == "SCALED_IMU":
                c["imu"] = (msg.xacc, msg.yacc, msg.zacc,
                            msg.xgyro, msg.ygyro, msg.zgyro)

            # setpoint check
            elif mtype == "POSITION_TARGET_LOCAL_NED":
                # check that ardupilot is not overriding the setpoint
                c["echoed_bitmask"] = msg.type_mask
                if c["sent_wall"] is not None:
                    c["bitmask_callback_dt"] = time.time() - c["sent_wall"]

            # battery check
            elif mtype == "SYS_STATUS":
                c["batt_voltage"] = (
                    msg.voltage_battery / 1000 if msg.voltage_battery != 65535 else NAN)   # convert mV to V

                c["batt_current"] = (
                    msg.current_battery / 100 if msg.current_battery != -1 else NAN)      # convert cA to A

                c["batt_rem_percent"] = (
                    msg.battery_remaining if msg.battery_remaining != -1 else NAN)

            # GPS health check
            elif mtype == "GPS_RAW_INT":
                c["gps_fix_type"] = msg.fix_type
                c["sat_count"] = msg.satellites_visible
                c["hdop"] = msg.eph / 100 if msg.eph != 65535 else NAN

            # EKF health check
            elif mtype == "EKF_STATUS_REPORT":
                c["ekf_flags"] = msg.flags
                c["ekf_pos_horiz_var"] = msg.pos_horiz_variance
                c["ekf_pos_vert_var"] = msg.pos_vert_variance
                c["ekf_vel_var"] = msg.velocity_variance

            # drain the RC channels
            elif mtype == "RC_CHANNELS":
                c["rc"] = [getattr(msg, f"chan{i}_raw")
                           for i in range(1, 9)]  # PWM us

            # command acknowledgement check
            elif mtype == "COMMAND_ACK":
                c["last_cmd"] = msg.command
                c["last_cmd_result"] = msg.result

            # event status check
            elif mtype == "STATUSTEXT":
                self.ew.writerow(
                    [f"{time.time():.4f}", c["fc_time_boot_ms"], msg.severity, msg.text])
                self.ef.flush()

    def log(self, t, x, p_ref, v_ref, u, *,
            yaw_ref=0, yaw_rate_ref=0,
            payload_p_ref=None, payload_v_ref=None,
            payload_alpha=None, payload_alphadot=None):
        """
        Call once per control tick, AFTER send_accel/note_sent.
          t: loop time since start (control loop's `t`)
          x: ENU state used by the controller; x[0:3]=pos, x[3:6]=vel
          p_ref,v_ref: drone ENU position/velocity references
          u: control output (ux,uy,uz), same frame you pass to the controller (usually the acceleration setpoint)
          payload_*: optional, None until the ArUco/KF payload tracker is online
        """
        # what time is it?
        now = time.time()
        now_m = time.monotonic()  # time goes FORWARD
        loop_dt = NAN if self._last_m is None else now_m - self._last_m
        self._last_m = now_m
        ctrl_freq = 1/loop_dt if loop_dt and loop_dt > 0 else NAN
        self._n += 1

        # initialize columns from the cache
        c = self.cache
        p, v = x[0:3], x[3:6]
        imu = c["imu"]
        pl_pr = payload_p_ref if payload_p_ref is not None else (NAN, NAN, NAN)
        pl_vr = payload_v_ref if payload_v_ref is not None else (NAN, NAN, NAN)
        pl_a = payload_alpha if payload_alpha is not None else (NAN, NAN)
        pl_ad = payload_alphadot if payload_alphadot is not None else (
            NAN, NAN)
        hb_age = now - \
            c["last_hb_wall"] if c["last_hb_wall"] is not None else NAN

        # build dictionary
        row = {
            # time
            "wall_time": f"{now:.4f}",
            "cur_time": f"{t:.4f}",
            "cur_loop_dt": f"{loop_dt:.5f}",
            "cur_ctrl_freq": f"{ctrl_freq:.2f}",
            "loop_count": self._n,
            "fc_time_boot_ms": c["fc_time_boot_ms"],
            "hb_age_s": f"{hb_age:.3f}",
            # mode
            "sent_mode": c["sent_mode"],
            "echoed_mode": c["echoed_mode"],
            "armed": int(c["armed"]),
            # drone
            "drone_px_ref": p_ref[0], "drone_py_ref": p_ref[1], "drone_pz_ref": p_ref[2],
            "drone_vx_ref": v_ref[0], "drone_vy_ref": v_ref[1], "drone_vz_ref": v_ref[2],
            # payload
            "payload_px_ref": pl_pr[0], "payload_py_ref": pl_pr[1], "payload_pz_ref": pl_pr[2],
            "payload_vx_ref": pl_vr[0], "payload_vy_ref": pl_vr[1], "payload_vz_ref": pl_vr[2],
            "payload_alpha_x": pl_a[0], "payload_alpha_y": pl_a[1],
            "payload_alphadot_x": pl_ad[0], "payload_alphadot_y": pl_ad[1],
            # control input
            "ux": u[0], "uy": u[1], "uz": u[2],
            "yaw_ref": yaw_ref, "yaw_rate_ref": yaw_rate_ref,
            # drone actual
            "drone_px_meas": p[0], "drone_py_meas": p[1], "drone_pz_meas": p[2],
            "drone_vx_meas": v[0], "drone_vy_meas": v[1], "drone_vz_meas": v[2],
            "drone_ax_meas": imu[0] , "drone_ay_meas": imu[1], "drone_az_meas": imu[2],
            "drone_gyrox_meas": imu[3], "drone_gyroy_meas": imu[4], "drone_gyroz_meas": imu[5],
            # drone orientation
            "drone_roll": c["roll"], "drone_pitch": c["pitch"],
            "drone_yaw": c["yaw"], "drone_yaw_rate": c["yaw_rate"],
            # bitmask
            "sent_setpoint_bitmask": c["sent_bitmask"],
            "echoed_setpoint_bitmask": c["echoed_bitmask"],
            "setpoint_bitmask_callback_dt": c["bitmask_callback_dt"],
            # battery
            "batt_voltage": c["batt_voltage"], "batt_current": c["batt_current"],
            "batt_rem_percent": c["batt_rem_percent"],
            # GPS
            "GPS_fix_type": c["gps_fix_type"], "sat_count": c["sat_count"], "HDOP": c["hdop"],
            # EKF
            "ekf_flags": c["ekf_flags"], "ekf_pos_horiz_var": c["ekf_pos_horiz_var"],
            "ekf_pos_vert_var": c["ekf_pos_vert_var"], "ekf_vel_var": c["ekf_vel_var"],
        }

        # RC channel values
        for i in range(8):
            row[f"ch{i + 1}"] = c["rc"][i]

        # add to csv file
        self.w.writerow([row.get(col, "") for col in COLUMNS])

        # put logging on queue + writer thread
        self.f.flush()
        if self._n % 100 == 0:  # approximately 2s at 50 Hz probably
            os.fsync(self.f.fileno())

    def close(self):
        """
        close the logger and sync to disk
        """
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()
        self.ef.flush()
        self.ef.close()
