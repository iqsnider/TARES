"""
Run the payload EKF on a recorded flight and produce:
  1. a static 3D ENU plot of drone / measured payload / EKF-estimated payload
  2. a time-series comparison of the EKF against the per-frame camera
     measurement -- swing angles and payload yaw -- with occlusion shading
     and the 2-sigma band
  3. with --cam, the camera recording written back out as an .avi with the
     estimated payload position drawn on every frame

The pose CSV has no wall clock, so the offset between the two logs comes off
the session (sessions.toml).
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sim.estimation.calculate_payload_position as payload
import sim.estimation.ekf as ekfm
import sim.estimation.pre_process as pp
from sim.plotting import (TimeSlider, configure_plot_style,
                          C_REF, C_DRONE, C_PAYLOAD)
import catalog
from run_on_log import build_inputs, group_frames, make_ekf, run_full

configure_plot_style()   # shared serif / Computer Modern theme

C_EST = "#009E73"
CALIB_FILE = (Path(__file__).resolve().parents[1] /
              "src/payload_tracking/camera_calibration/calibration.json")

BGR_EST = (60, 255, 80)


def direct_measurement(frames, inp, offset):
    """
    Per-frame swing angles straight from the PnP chain, no filtering.
    This is the orange scatter the EKF gets compared against. It is not
    truth -- it carries the same attitude error plus PnP depth noise.

    The same pre-processing the filter is fed, so the two are exactly
    comparable.
    """
    t_fl = inp["t"]
    rows = []
    for t_cam, g, det in frames:
        t = t_cam - offset
        if t < t_fl[0] or t > t_fl[-1]:
            continue
        phi = np.interp(t, t_fl, inp["phi"])
        theta = np.interp(t, t_fl, inp["theta"])
        psi = np.interp(t, t_fl, inp["psi"])
        m = pp.swing_angles(g, ekfm.T_IB_fn(phi, theta, psi))
        if m is None:
            continue
        rows.append((t_cam, *m, len(det)))
    return pd.DataFrame(rows, columns=["t_cam", "alpha_x", "alpha_y",
                                       "psi_p", "n"])


def estimated_payload_enu(records, fl, L_m):
    """p_payload = p_drone + L_m * q_I, drone position interpolated.

    Returns (t_flight, ENU) so the caller can put the estimate on a timeline.
    """
    t = fl.cur_time.to_numpy()
    tf = np.array([r["t_flight"] for r in records])
    E = np.interp(tf, t, fl.drone_px_meas.to_numpy())
    N = np.interp(tf, t, fl.drone_py_meas.to_numpy())
    U = np.interp(tf, t, fl.drone_pz_meas.to_numpy())
    q = np.array([r["q_I"] for r in records])
    return tf, np.c_[E, N, U] + L_m * q


def analyse(session, verbose=True):
    """Run the payload EKF over a session. Returns what the plots need.

    Everything the filter depends on that is not in the data itself -- the
    clock offset, the tether lengths -- comes off the session (i.e. out of
    sessions.toml), so there are no paths or constants here.
    """
    if not session.has_camera:
        raise SystemExit(f"session {session.id} has no camera data")

    offset = session.pose_offset
    inp = build_inputs(session.fl)
    frames = group_frames(session.poses)

    records = run_full(frames, inp, offset)
    meas_df = direct_measurement(frames, inp, offset)
    if verbose:
        print(f"{len(records)} filter steps, "
              f"{sum(r['n'] > 0 for r in records)} with a measurement")
    R = summarise(records, meas_df) if verbose else None

    est_t, est = estimated_payload_enu(records, session.fl, session.L_marker)
    pdf = payload.get_payload_ENU_from_data(session.pose, session.flight,
                                            time_offset=offset)
    return dict(records=records, meas_df=meas_df, R=R, est_t=est_t, est=est,
                pdf=pdf, offset=offset)


def summarise(records, meas_df):
    """Print the numbers worth checking after every run."""
    R = pd.DataFrame([{k: v for k, v in r.items()
                       if k in ("t_cam", "n",
                                "alpha_x", "alpha_y", "psi_p",
                                "sigma_alpha_x", "sigma_alpha_y",
                                "sigma_psi_p")}
                      for r in records])
    sel = R.n > 0
    mi = np.interp(R.t_cam, meas_df.t_cam, meas_df.alpha_x)
    mj = np.interp(R.t_cam, meas_df.t_cam, meas_df.alpha_y)
    gaps = np.diff(R.t_cam[sel].to_numpy())
    print(f"  alpha_x RMS vs direct measurement : "
          f"{np.rad2deg(np.sqrt(np.mean((R.alpha_x[sel]-mi[sel])**2))):.3f} deg")
    print(f"  alpha_y RMS vs direct measurement : "
          f"{np.rad2deg(np.sqrt(np.mean((R.alpha_y[sel]-mj[sel])**2))):.3f} deg")
    print(f"  mean 1-sigma alpha                : "
          f"{np.rad2deg(R.sigma_alpha_x[sel].mean()):.3f}, "
          f"{np.rad2deg(R.sigma_alpha_y[sel].mean()):.3f} deg")
    print(f"  measurement gaps                  : median {np.median(gaps):.3f} s, "
          f"max {gaps.max():.2f} s")
    return R


# --------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------
def _set_equal_3d(ax, X, Y, Z):
    r = max(np.ptp(X), np.ptp(Y), np.ptp(Z)) / 2 or 1.0
    cx, cy, cz = (X.max()+X.min())/2, (Y.max()+Y.min())/2, (Z.max()+Z.min())/2
    ax.set_xlim(cx-r, cx+r)
    ax.set_ylim(cy-r, cy+r)
    ax.set_zlim(cz-r, cz+r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def plot_3d(fl, payload_df, est_t, est_enu, save=None, slider=True):
    """Drone/payload/EKF paths in ENU.

    `est_t` is the flight time of each row of `est_enu`. With `slider` the
    on-screen figure gets a time slider that plays the run back; the saved PNG
    is written first, so the file always holds the whole flight.
    """
    t = fl.cur_time.to_numpy()
    E = fl.drone_px_meas.to_numpy()
    N = fl.drone_py_meas.to_numpy()
    U = fl.drone_pz_meas.to_numpy()
    pE = payload_df.payload_e.to_numpy()
    pN = payload_df.payload_n.to_numpy()
    pU = payload_df.payload_u.to_numpy()

    fig = plt.figure(figsize=(9.5, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    drone_ln, = ax.plot(E, N, U, color=C_DRONE, lw=2.0, label="Drone")
    ref_ln = None
    if {"drone_px_ref", "drone_py_ref", "drone_pz_ref"} <= set(fl.columns):
        ref_ln, = ax.plot(fl.drone_px_ref, fl.drone_py_ref, fl.drone_pz_ref,
                          color=C_REF, lw=1.6, linestyle=(0, (5, 4)),
                          label="Drone reference")

    finite = np.flatnonzero(np.isfinite(pE))
    tethers = [ax.plot([E[k], pE[k]], [N[k], pN[k]], [U[k], pU[k]],
                       color="#5F6368", lw=0.9, alpha=0.35,
                       label="Tether (snapshots)" if j == 0 else None)[0]
               for j, k in enumerate(finite[::25])]

    meas_ln, = ax.plot(pE, pN, pU, color=C_PAYLOAD, lw=1.4, alpha=0.85,
                       label="Payload (camera measurement)")
    est_ln, = ax.plot(est_enu[:, 0], est_enu[:, 1], est_enu[:, 2],
                      color=C_EST, lw=2.0, label="Payload (EKF estimate)")

    # the tether where the slider is; static until a slider is attached
    live_tether, = ax.plot([E[-1], est_enu[-1, 0]], [N[-1], est_enu[-1, 1]],
                           [U[-1], est_enu[-1, 2]], color="#3C4043", lw=1.8,
                           label="Tether (at $t$)")

    start_pt = ax.scatter([E[0]], [N[0]], [U[0]], color="#2CA02C",
                          edgecolors="white", linewidths=0.8, marker="^",
                          s=90, depthshade=False, label="Start")
    head, = ax.plot([E[-1]], [N[-1]], [U[-1]], marker="s", ms=8,
                    color="#222222", mec="white", mew=0.8, linestyle="none",
                    label="End")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Up [m]")
    ax.set_title("Experiment: Drone and Payload, "
                 "Camera Measurement vs EKF")
    _set_equal_3d(ax,
                  np.concatenate([E, pE[finite], est_enu[:, 0]]),
                  np.concatenate([N, pN[finite], est_enu[:, 1]]),
                  np.concatenate([U, pU[finite], est_enu[:, 2]]))

    # same running order as the simulation figure, which draws its artists in
    # a different sequence than this one does
    order = [ref_ln, drone_ln, meas_ln, est_ln,
             tethers[0] if tethers else None, live_tether, start_pt, head]
    ax.legend(handles=[h for h in order if h is not None],
              fontsize=13.5, loc="upper left")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120)
        print("wrote", save)

    if slider:
        s = TimeSlider(fig, t, label="flight time [s]")
        s.line(drone_ln, np.c_[E, N, U])
        s.line(meas_ln, np.c_[pE, pN, pU], payload_df.cur_time.to_numpy())
        s.line(est_ln, est_enu, est_t)
        s.marker(head, np.c_[E, N, U])
        # to the EKF estimate, the one payload track with no detection gaps
        s.span(live_tether, np.c_[E, N, U], est_enu, t_b=est_t)
        s.group(tethers, t[finite[::25]])
        if ref_ln is not None:
            s.line(ref_ln, fl[["drone_px_ref", "drone_py_ref",
                               "drone_pz_ref"]].to_numpy())
    return fig


def _break_wraps(deg, jump=180.0):
    """NaN the sample after a +-180 wrap, so the line does not draw a vertical
    stripe across the panel. Only psi_p wraps."""
    deg = np.asarray(deg, float).copy()
    deg[np.r_[False, np.abs(np.diff(deg)) > jump]] = np.nan
    return deg


def plot_timeseries(R, meas_df, save=None):
    """EKF vs per-frame measurement, with the 2-sigma band."""
    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    # psi_p is a measured state now, so it gets a panel like the swing angles
    for k, (nm, lbl, sg) in enumerate(
            [("alpha_x", r"$\alpha_x$", "sigma_alpha_x"),
             ("alpha_y", r"$\alpha_y$", "sigma_alpha_y"),
             ("psi_p", r"$\psi_P$", "sigma_psi_p")]):
        a = axs[k]
        y = _break_wraps(np.rad2deg(R[nm]))
        sig = np.rad2deg(R[sg])
        a.plot(meas_df.t_cam, np.rad2deg(meas_df[nm]), ".", ms=4,
               color="tab:orange", label="per-frame camera measurement")
        a.plot(R.t_cam, y, "-", lw=1.6, color="tab:red",
               label="EKF estimate")
        a.fill_between(R.t_cam, y - 2*sig, y + 2*sig,
                       color="tab:red", alpha=0.2,
                       label=r"EKF $2\sigma$")
        a.set_ylabel(f"{lbl} [deg]")
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="upper right")

    axs[3].plot(R.t_cam, R.n, "k.", ms=3)
    axs[3].set_ylabel("markers seen")
    axs[3].set_yticks([0, 1, 2, 3])
    axs[3].set_xlabel("camera time [s]")
    axs[3].grid(alpha=0.3)
    axs[0].set_title("Payload swing: EKF vs direct measurement")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=115)
        print("wrote", save)
    return fig


# --------------------------------------------------------------------------
# camera overlay
# --------------------------------------------------------------------------
OVERLAY_SUFFIX = "_ekf_overlay.avi"


def find_recording(session):
    """The video the recorder wrote beside a session's poses.csv, if kept.

    Only the pose folder itself is searched, not below it: what sits in an
    output subfolder is something a script drew, not the flight. Overlays this
    one wrote earlier are skipped too, so a re-run cannot draw on its own
    output.
    """
    vids = sorted(p for ext in ("*.avi", "*.mp4", "*.mkv")
                  for p in session.pose.parent.glob(ext)
                  if not p.name.endswith(OVERLAY_SUFFIX))
    return vids[0] if vids else None


def _intrinsics():
    """The camera matrix and distortion the poses were solved with.

    Used unscaled even though the recording is larger than the calibration
    resolution: the recorder solved PnP with this matrix on these frames, and
    reprojecting its own points lands within a pixel of the u_px/v_px it
    logged, so this is the mapping the video is actually in.
    """
    with open(CALIB_FILE) as f:
        calib = json.load(f)
    return (np.array(calib["mtx"], dtype=float),
            np.array(calib["dist"], dtype=float))


def _label(frame, c, lines, gap=14, scale=1.15, thick=3):
    """Write `lines` up and to the right of the point at `c`.

    Nudged back inside the frame when the payload swings near an edge, so the
    readout never runs off the picture.
    """
    step = int(34*scale)
    x = c[0] + gap
    y = c[1] - gap - step*(len(lines) - 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    width = max(cv2.getTextSize(s, font, scale, thick)[0][0] for s in lines)
    x = min(x, frame.shape[1] - width - 4)
    y = max(y, step)

    for k, text in enumerate(lines):
        at = (x, y + k*step)
        cv2.putText(frame, text, at, font, scale, (0, 0, 0), thick + 3,
                    cv2.LINE_AA)
        cv2.putText(frame, text, at, font, scale, BGR_EST, thick, cv2.LINE_AA)


def overlay_video(session, records, save=None, n_sigma=2):
    """Redraw the recording with the EKF payload estimate on each frame.

    A green dot marks where the filter believes the payload is and the shaded
    ellipse around it is its `n_sigma` position uncertainty, both projected
    into the camera by `EKF.estimate_to_px_coords`. The swing velocity is
    written beside it. Frames the filter never reached are copied through
    untouched.
    """
    src = find_recording(session)
    if src is None:
        raise SystemExit(f"session {session.id} has no recording next to "
                         f"{session.pose.name}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    out = Path(save) if save else src.with_name(session.id + OVERLAY_SUFFIX)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"MJPG"),
                             fps, (w, h))

    K, D = _intrinsics()
    filt = make_ekf(0, 0, 0, 0, 0, 0)
    drawn = {}
    for r in records:
        center, axes, angle = filt.estimate_to_px_coords(
            r["xi"], r["P"], r["T_IB"], K, D, n_sigma)
        # a payload swung out of the camera's half-space projects nowhere
        if np.isfinite(center).all() and np.isfinite(axes).all():
            drawn[r["frame"]] = (center, axes, angle,
                                 filt.estimate_to_swing_velocity(r["xi"]))

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        hit = drawn.get(i)
        if hit is not None:
            center, axes, angle, (v_x, v_y) = hit
            c = tuple(np.round(center).astype(int))
            ax = tuple(np.maximum(np.round(axes).astype(int), 1))
            shade = frame.copy()
            cv2.ellipse(shade, c, ax, angle, 0, 360, BGR_EST, -1)
            cv2.addWeighted(shade, 0.35, frame, 0.65, 0, frame)
            cv2.ellipse(frame, c, ax, angle, 0, 360, BGR_EST, 2)

            # the dot stays smaller than the band it sits in; at this range a
            # 2-sigma ellipse is only about ten pixels across
            cv2.circle(frame, c, 4, BGR_EST, -1)
            cv2.circle(frame, c, 4, (255, 255, 255), 1)
            _label(frame, c, (f"vE {v_x:+.2f} m/s", f"vN {v_y:+.2f} m/s"))
        writer.write(frame)
        i += 1

    cap.release()
    writer.release()
    print(f"wrote {out}  ({len(drawn)} of {i} frames carry an estimate)")
    return out


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("selector", nargs="?", default="latest.cam",
                    help="which session (see `uv run plot.py ls`)")
    ap.add_argument("--save", metavar="DIR", default=None,
                    help="write the figures here")
    args = ap.parse_args()

    session = catalog.resolve(args.selector)
    note = session.meta.get("note", "")
    print(f"{session.id}  {session.label}{'  -- ' + note if note else ''}")

    r = analyse(session)

    out = Path(args.save).expanduser() if args.save else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    plot_3d(session.fl, r["pdf"], r["est_t"], r["est"],
            save=out and out / f"{session.id}_ekf_3d.png")
    plot_timeseries(r["R"], r["meas_df"],
                    save=out and out / f"{session.id}_ekf_timeseries.png")

    plt.show()   # interactive, so the 3-D view can be panned
