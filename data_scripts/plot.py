"""Plot a flight by name instead of by path.

    uv run plot.py ls                     every session, newest last
    uv run plot.py ls 07232026            just that day
    uv run plot.py latest drone           the newest flight's log plots
    uv run plot.py 07232026.last ekf      the EKF on that day's last run
    uv run plot.py 07232026.114556 traj   one run, named in full

Dates are MMDDYYYY, the way the data folders are named, and a run is
date.time -- name both, so a coincidence between years cannot bite you.

Figures open on screen and are not written anywhere; pass --save DIR when you
actually want files. See catalog.py for how sessions are found and selected.
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

import catalog
from sim.plotting import configure_plot_style

configure_plot_style()   # shared serif / Computer Modern theme


def _out(save, name):
    """Path to write `name` to, or None when we are only showing."""
    if not save:
        return None
    save = Path(save).expanduser()
    save.mkdir(parents=True, exist_ok=True)
    return save / name


# ----------------------------------------------------------------------------
# what each kind draws
# ----------------------------------------------------------------------------
def kind_drone(s, save, cam=False, post=False):
    """Everything the flight log alone can show."""
    import drone_plot
    # only needed to draw the payload track, so a session with no onboard
    # EKF states doesn't have to carry a config_snapshot.json to be plotted
    L = s.config.get("TETHER_LEN") if catalog.has_logged_states(s.fl) else None
    drone_plot.plot_suite(s.fl, save=save and Path(save).expanduser(),
                          stem=s.id, L=L)


def kind_traj(s, save, cam=False, post=False):
    """3-D trajectory of the drone with the camera-measured payload."""
    import payload_plot
    import sim.estimation.pre_process as pp
    from sim.estimation.calculate_payload_position import (
        get_payload_ENU_from_data)
    if not s.has_camera:
        raise SystemExit(f"session {s.id} has no camera data; try `drone`")
    pdf = get_payload_ENU_from_data(
        s.pose, s.fl, time_offset=s.pose_offset,
        geom=pp.Geometry.from_snapshot(s.config),
        control_freq=s.config.get("CONTROL_FREQUENCY"))
    payload_plot.trajectory_plot_3d_with_payload(
        s.fl, pdf, save=_out(save, f"{s.id}_trajectory_3d.png"))


def kind_ekf(s, save, cam=False, post=False):
    """Run the payload EKF and show the 3-D and time-series comparisons.

    With `cam`, the recording is also written back out as an .mp4 carrying the
    estimated payload position.
    """
    import ekf_plots
    r = ekf_plots.analyze(s)
    ekf_plots.plot_3d(s.fl, r["pdf"], r["est_t"], r["est"],
                      save=_out(save, f"{s.id}_ekf_3d.png"),
                      from_log=r["from_log"])
    ekf_plots.plot_timeseries(r["R"], r["meas_df"],
                              save=_out(save, f"{s.id}_ekf_timeseries.png"),
                              from_log_cov=r["from_log_cov"])
    if cam:
        kind_overlay(s, save, post=post)


def kind_overlay(s, save, cam=False, post=False):
    """The recording with the payload estimate and the flown reference on it.

    Where the aircraft flew the filter itself, its logged states are what goes
    on the picture: that is the estimate the controller actually steered on,
    and re-running the filter offline would draw a different one. Only runs
    that predate the onboard filter are estimated here.

    With `post`, the filter is re-run offline on the geometry and tuning in the
    session's own config_snapshot.json. That is what you want once you have
    calibrated something the flight itself did not know: editing the snapshot
    afterwards cannot move states that were already logged, so it would
    otherwise only move the drawing, not the estimate.
    """
    import ekf_plots
    if post or not catalog.has_logged_states(s.fl):
        records = ekf_plots.analyze(s, verbose=False)["records"]
    else:
        records = ekf_plots.logged_records(s)

    # a post-hoc run gets its own name, so it never overwrites the estimate the
    # aircraft actually flew on
    stem = f"{s.id}_posthoc" if post else s.id
    out = _out(save, stem + ekf_plots.OVERLAY_SUFFIX)
    if out is None and post:
        out = s.pose.parent / (stem + ekf_plots.OVERLAY_SUFFIX)
    ekf_plots.overlay_video(s, records, save=out)


KINDS = {"drone": kind_drone, "traj": kind_traj, "ekf": kind_ekf,
         "overlay": kind_overlay}


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("selector", nargs="?", default="latest",
                    help="which session: latest, 07232026.last, 07232026.114556")
    # not `choices`: after `ls` this slot holds a selector, not a kind
    ap.add_argument("kind", nargs="?", default=None,
                    help="what to draw: " + ", ".join(sorted(KINDS)))
    ap.add_argument("--save", metavar="DIR", default=None,
                    help="write PNGs here instead of only showing them")
    ap.add_argument("--cam", action="store_true",
                    help="ekf only: also write the recording back out as an "
                         ".mp4 with the estimated payload drawn on it")
    ap.add_argument("--post", action="store_true",
                    help="overlay only: re-run the EKF offline on the session's "
                         "config_snapshot.json, instead of drawing the states "
                         "the aircraft logged")
    ap.add_argument("--root", default=None, help="override the data root")
    args = ap.parse_args(argv)

    if args.root:
        catalog.DATA_ROOT = Path(args.root).expanduser()

    # `plot.py ls [selector]` lists instead of drawing
    if args.selector == "ls":
        try:
            pool = catalog.select(args.kind or "all")
        except (FileNotFoundError, LookupError) as e:
            print(e, file=sys.stderr)
            return 1
        print(catalog.format_table(pool))
        strays = catalog.unpaired_poses()
        if strays:
            print(f"\n{len(strays)} camera file(s) with no timestamp, so not "
                  f"attached to any flight:")
            for p in strays:
                print(f"    {p.relative_to(catalog.DATA_ROOT)}")
        return 0

    if args.kind is None:
        ap.error("say what to draw: " + ", ".join(sorted(KINDS)))
    if args.kind not in KINDS:
        ap.error(f"unknown kind {args.kind!r}; choose from "
                 + ", ".join(sorted(KINDS)))
    if args.cam and args.kind != "ekf":
        ap.error("--cam only applies to `ekf`")
    if args.post and args.kind not in ("overlay", "ekf"):
        ap.error("--post only applies to `overlay` and `ekf --cam`")

    try:
        s = catalog.resolve(args.selector)
    except (FileNotFoundError, LookupError) as e:
        print(e, file=sys.stderr)
        return 1
    note = s.meta.get("note", "")
    print(f"{s.id}  {s.label}{'  -- ' + note if note else ''}")
    print(f"   log  {s.flight}")
    if s.has_camera:
        print(f"   cam  {s.pose}   offset {s.pose_offset:+.2f} s")

    KINDS[args.kind](s, args.save, cam=args.cam, post=args.post)

    if args.save:
        print(f"figures written to {args.save}/")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
