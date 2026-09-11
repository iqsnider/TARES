"""
Read the flight controller parameters that set how fast a commanded
acceleration turns into a lean angle. Run on the field laptop with the FC
connected, nothing is written.
"""
import sys
import comms.common as comms
from pymavlink import mavutil

PARAMS = [
    "GUID_OPTIONS", "PSC_JERK_XY", "PSC_NE_JERK", "PSC_ANGLE_MAX", "ANGLE_MAX",
    "PSC_VELXY_FLTE", "PSC_VELXY_FLTD", "PSC_POSXY_P", "PSC_VELXY_P",
    "WPNAV_SPEED", "WPNAV_ACCEL", "WPNAV_JERK",
    "ATC_INPUT_TC", "ATC_ACCEL_R_MAX", "ATC_ACCEL_P_MAX",
    "ATC_RATE_R_MAX", "ATC_RATE_P_MAX", "ATC_ANG_RLL_P", "ATC_ANG_PIT_P",
    "ATC_RAT_RLL_P", "ATC_RAT_RLL_I", "ATC_RAT_RLL_D", "ATC_RAT_RLL_FLTT",
    "ATC_RAT_PIT_P", "ATC_RAT_PIT_I", "ATC_RAT_PIT_D", "ATC_RAT_PIT_FLTT",
    "ATC_SLEW_YAW", "MOT_THST_HOVER", "INS_GYRO_FILTER", "INS_ACCEL_FILTER",
]

if __name__ == "__main__":
    connection = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
    m = comms.connect(connection, 115200)
    for name in PARAMS:
        m.mav.param_request_read_send(m.target_system, m.target_component,
                                      name.encode(), -1)
        msg = None
        for _ in range(20):
            msg = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=0.5)
            if msg is not None and msg.param_id.strip("\x00") == name:
                break
            msg = None
        print("%-18s %s" % (name, "%.4g" % msg.param_value if msg else "no reply"))
