"""
test for the color ring tracker.
"""

# autonomy research imports
import Prm.config as config
from payload_tracking.color_track import ColorCircleRecorder

# logging
from datetime import datetime


if __name__ == '__main__':
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_dir = f"data/color_track_{stamp}"
    video_out = f"{data_dir}/recording.avi"
    circles_out = f"{data_dir}/circles.csv"

    recorder = ColorCircleRecorder(
        circle_diameter_m=config.CIRCLE_DIAMETER,
        band_m=config.CIRCLE_BAND,
        hue=config.CIRCLE_HUE,
        hue_width=config.CIRCLE_HUE_WIDTH,
        sat_min=config.CIRCLE_SAT_MIN,
        val_min=config.CIRCLE_VAL_MIN,
        min_area_px=config.CIRCLE_MIN_AREA_PX,
        min_coverage_deg=config.CIRCLE_MIN_COVERAGE_DEG,
        video_out=video_out,
        csv_out=circles_out,
        capture_fps=config.CAM_FPS,
        frame_stride=config.CAM_STRIDE,
        gain=config.CAM_GAIN,
        exposure_abs=config.CAM_EXP_ABS,
        preview_port=config.CAM_PREVIEW_PORT,
        print_every=config.CAM_FPS)

    recorder.run()
