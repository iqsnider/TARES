"""
Redraw a session's recording with the swing re-estimated behind a tight
range gate.

    uv run data_scripts/render_gated_overlays.py 09112026.100158 ...

The tracker flew a +-80% range gate, so a ring seen at a metre or two on a
6.2 m tether reached the onboard filter and threw the estimate. This drops
those detections and replays the filter, which is what `icra_plots` draws.
Frames past the end of the flight log get no estimate rather than one built
on a clamped attitude. Output goes beside the recording as
<id>_gated_ekf_overlay.mp4, leaving the as-flown overlay alone.
"""

import sys
import time

import numpy as np

import catalog
import ekf_plots
import sim.estimation.ekf as ekfm
import sim.estimation.pre_process as pp
from icra_plots import STICK_RANGE_TOL
from run_on_log import build_inputs, group_circles, run_full

SUFFIX = "_gated" + ekf_plots.OVERLAY_SUFFIX


def gated_records(s):
    """
    The filter replayed with the false ring detections dropped
    """
    fl = s.fl
    cfg = s.config
    L = cfg["TETHER_LEN"]

    poses = s.poses.copy()
    rng = poses.range_m.to_numpy(float)
    seen = np.isfinite(poses.u_px.to_numpy(float))
    bad = seen & (np.abs(rng - L) > STICK_RANGE_TOL*L)
    poses.loc[bad, ["u_px", "v_px", "radius_px", "coverage_deg", "n_px",
                    "x", "y", "z", "range_m"]] = np.nan

    records = run_full(group_circles(poses), build_inputs(fl), s.pose_offset,
                       geom=pp.Geometry.from_snapshot(cfg),
                       hold=ekf_plots.held_at(ekf_plots.hold_windows(fl)),
                       L=L, source=ekfm.SOURCE_COLOR,
                       **ekf_plots.ekf_tuning(cfg))

    t = fl.cur_time.to_numpy()
    inside = [r for r in records if t[0] <= r["t_flight"] <= t[-1]]

    return inside, int(bad.sum()), len(records) - len(inside)


def render(sel):
    s = catalog.resolve(sel)
    t0 = time.time()
    records, dropped, outside = gated_records(s)
    out = s.pose.parent/(s.id + SUFFIX)
    print(f"{s.id}  {s.label}: {dropped} detections gated, "
          f"{outside} frames past the log, {len(records)} records")
    ekf_plots.overlay_video(s, records, save=out)
    print(f"   {time.time() - t0:.0f}s")


if __name__ == '__main__':
    for sel in sys.argv[1:]:
        render(sel)
