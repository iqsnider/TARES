"""Plot a flight by name instead of by path.

    uv run plot.py ls                 every session, newest last
    uv run plot.py ls 0723            just that day
    uv run plot.py latest drone       the newest flight's log plots
    uv run plot.py 0723.last ekf      the EKF on the last run of 23 Jul
    uv run plot.py 114556 traj        3-D trajectory, by timestamp fragment

Figures open on screen and are not written anywhere; pass --save DIR when you
actually want files. See catalog.py for how sessions are found and selected.
"""
import argparse
import sys
import tempfile
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
def kind_drone(s, save):
    """Everything the flight log alone can show."""
    import drone_plot
    drone_plot.plot_suite(s.fl, save=save and Path(save).expanduser(),
                          stem=s.id)


def kind_traj(s, save):
    """3-D trajectory of the drone with the camera-measured payload."""
    import payload_plot
    from sim.estimation.calculate_payload_position import (
        get_payload_ENU_from_data)
    if not s.has_camera:
        raise SystemExit(f"session {s.id} has no camera data; try `drone`")
    pdf = get_payload_ENU_from_data(s.pose, s.flight,
                                    time_offset=s.pose_offset)
    payload_plot.trajectory_plot_3d_with_payload(
        s.fl, pdf, save=_out(save, f"{s.id}_trajectory_3d.png"))


def kind_ekf(s, save):
    """Run the payload EKF and show the 3-D and time-series comparisons."""
    import ekf_plots
    r = ekf_plots.analyse(s)
    ekf_plots.plot_3d(s.fl, r["pdf"], r["est_t"], r["est"],
                      save=_out(save, f"{s.id}_ekf_3d.png"))
    ekf_plots.plot_timeseries(r["R"], r["meas_df"], r["offset"],
                              save=_out(save, f"{s.id}_ekf_timeseries.png"))


def kind_video(s, save):
    """Camera image plane animation. Always a file -- there is no other form."""
    import ekf_plots
    r = ekf_plots.analyse(s)
    path = _out(save, f"{s.id}_camera_view.mp4")
    if path is None:
        path = Path(tempfile.mkdtemp(prefix="tares_")) / f"{s.id}_camera.mp4"
    ekf_plots.animate_camera(r["records"], path)
    ekf_plots.open_file(path)


KINDS = {"drone": kind_drone, "traj": kind_traj,
         "ekf": kind_ekf, "video": kind_video}


# ----------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("selector", nargs="?", default="latest",
                    help="which session: latest, 0723.last, 114556, ...")
    # not `choices`: after `ls` this slot holds a selector, not a kind
    ap.add_argument("kind", nargs="?", default=None,
                    help="what to draw: " + ", ".join(sorted(KINDS)))
    ap.add_argument("--save", metavar="DIR", default=None,
                    help="write PNGs here instead of only showing them")
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

    KINDS[args.kind](s, args.save)

    if args.save:
        print(f"figures written to {args.save}/")
    plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
