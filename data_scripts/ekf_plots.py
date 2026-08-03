"""
Run the payload EKF on a recorded flight and produce:
  1. a static 3D ENU plot of drone / measured payload / EKF-estimated payload
  2. a time-series comparison of the EKF against the per-frame camera
     measurement, with occlusion shading and the 2-sigma band
  3. an animation of the camera-frame line of sight showing the measured
     marker bearings against the EKF's predicted ones, with a 1-sigma
     uncertainty ellipse

The pose CSV has no wall clock, so the offset between the two logs is
estimated by minimising NIS (see run_on_log.py) rather than guessed.
"""
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.animation as animation

import sim.config as config
import sim.estimation.calculate_payload_position as payload
import sim.estimation.ekf as ekfm
from sim.plotting import TimeSlider, configure_plot_style
import catalog
from run_on_log import build_inputs, group_frames, make_params

configure_plot_style()   # shared serif / Computer Modern theme

G = 9.80665
IDS = [config.LEFT_MARKER_ID, config.CENTER_MARKER_ID, config.RIGHT_MARKER_ID]
ID_NAME = {config.LEFT_MARKER_ID: "LEFT 232",
           config.CENTER_MARKER_ID: "CENTER 245",
           config.RIGHT_MARKER_ID: "RIGHT 233"}
ID_COLOR = {config.LEFT_MARKER_ID: "tab:blue",
            config.CENTER_MARKER_ID: "tab:green",
            config.RIGHT_MARKER_ID: "tab:orange"}


def open_file(path):
    """Open a file with the OS default application."""
    path = str(path)
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        elif sys.platform.startswith("win"):
            os.startfile(path)                                  # noqa: S606
        else:
            subprocess.run(["xdg-open", path], check=False)
    except Exception as e:                                      # noqa: BLE001
        print(f"could not open {path}: {e}")


# --------------------------------------------------------------------------
# filter run
# --------------------------------------------------------------------------
def run_full(frames, inp, params, offset):
    """Same recursion as run_on_log.run, but also records the bearings."""
    t_fl = inp["t"]
    xi = P = None
    t_prev = None
    out = []
    for t_cam, meas, det, g in frames:
        t = t_cam - offset
        if t < t_fl[0] or t > t_fl[-1]:
            continue
        phi = np.interp(t, t_fl, inp["phi"])
        theta = np.interp(t, t_fl, inp["theta"])
        psi = np.interp(t, t_fl, inp["psi"])
        f_I = np.array([np.interp(t, t_fl, inp["f_I"][:, k]) for k in range(3)])

        if xi is None:
            if not meas:
                continue
            p_C_payload = payload.get_payload_center_in_camera_frame(g)
            C_CM, _ = cv2.Rodrigues(det[["rx", "ry", "rz"]].iloc[0].to_numpy())
            xi, P = ekfm.initial_state(p_C_payload, phi, theta, psi, C_CM=C_CM)
            t_prev = t_cam
            continue

        dt = min(max(t_cam - t_prev, 1e-3), 0.5)
        t_prev = t_cam
        xi, P, info = ekfm.ekf(xi, P, f_I, dt, params, measurements=meas,
                               phi=phi, theta=theta, psi=psi)

        C_BI = ekfm.C_IB_enu(phi, theta, psi).T
        pred = {}
        for mid in IDS:
            h_j, H_j, p_C_j = ekfm.marker_prediction(
                xi, payload.MARKER_OFFSET[mid], C_BI, params)
            # a marker behind the camera still has a bearing, a mirrored one
            pred[mid] = h_j if p_C_j[2] > 0 else None
        h_c, H_c, p_C_c = ekfm.marker_prediction(xi, 0.0, C_BI, params)
        S_c = H_c @ P @ H_c.T if p_C_c[2] > 0 else None
        if p_C_c[2] <= 0:
            h_c = None

        # what the filter was handed, as unit vectors
        b_meas = {}
        for mid, bx, by, bz in meas:
            b = np.array([bx, by, bz], float)
            b_meas[int(mid)] = b/np.linalg.norm(b)

        out.append(dict(t_cam=t_cam, t_flight=t, n=info["n_markers"],
                        nis=info["nis"],
                        alpha_x=xi[ekfm.IX_ALPHA_X],
                        alpha_y=xi[ekfm.IX_ALPHA_Y],
                        alpha_dot_x=xi[ekfm.IX_ALPHA_DOT_X],
                        alpha_dot_y=xi[ekfm.IX_ALPHA_DOT_Y],
                        psi_p=xi[ekfm.IX_PSI_P],
                        sigma_alpha_x=math.sqrt(P[ekfm.IX_ALPHA_X,
                                                 ekfm.IX_ALPHA_X]),
                        sigma_alpha_y=math.sqrt(P[ekfm.IX_ALPHA_Y,
                                                 ekfm.IX_ALPHA_Y]),
                        q_I=ekfm.q_I(xi[ekfm.IX_ALPHA_X], xi[ekfm.IX_ALPHA_Y]),
                        pred=pred, pred_c=h_c, S_c=S_c, meas=b_meas))
    return out


def direct_measurement(frames, inp, offset):
    """
    Per-frame swing angles straight from the PnP chain, no filtering.
    This is the orange scatter the EKF gets compared against. It is not
    truth -- it carries the same attitude error plus PnP depth noise.
    """
    t_fl = inp["t"]
    t_BC_B = np.array([config.CAM_OFFSET_X, config.CAM_OFFSET_Y,
                       config.CAM_OFFSET_Z])
    rows = []
    for t_cam, meas, det, g in frames:
        if len(det) == 0:
            continue
        t = t_cam - offset
        if t < t_fl[0] or t > t_fl[-1]:
            continue
        p_C_payload = payload.get_payload_center_in_camera_frame(g)
        if p_C_payload is None:
            continue
        p_B = config.CAM_R @ p_C_payload + t_BC_B
        phi = np.interp(t, t_fl, inp["phi"])
        theta = np.interp(t, t_fl, inp["theta"])
        psi = np.interp(t, t_fl, inp["psi"])
        alpha_x, alpha_y = ekfm.alpha_from_q_I(
            ekfm.C_IB_enu(phi, theta, psi) @ p_B)
        rows.append((t_cam, alpha_x, alpha_y, len(det)))
    return pd.DataFrame(rows, columns=["t_cam", "alpha_x", "alpha_y", "n"])


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
    params = make_params(session.L_dyn, L_m=session.L_marker)
    inp = build_inputs(session.fl)
    frames = group_frames(session.poses)

    records = run_full(frames, inp, params, offset)
    meas_df = direct_measurement(frames, inp, offset)
    if verbose:
        print(f"{len(records)} filter steps, "
              f"{sum(r['n'] > 0 for r in records)} with a measurement")
    R = summarise(records, meas_df) if verbose else None

    est_t, est = estimated_payload_enu(records, session.fl, session.L_marker)
    pdf = payload.get_payload_ENU_from_data(session.pose, session.flight,
                                            time_offset=offset)
    return dict(records=records, meas_df=meas_df, R=R, est_t=est_t, est=est,
                pdf=pdf, offset=offset, params=params)


def summarise(records, meas_df):
    """Print the numbers worth checking after every run."""
    R = pd.DataFrame([{k: v for k, v in r.items()
                       if k in ("t_cam", "n", "nis", "alpha_x", "alpha_y",
                                "sigma_alpha_x", "sigma_alpha_y")}
                      for r in records])
    sel = R.n > 0
    mi = np.interp(R.t_cam, meas_df.t_cam, meas_df.alpha_x)
    mj = np.interp(R.t_cam, meas_df.t_cam, meas_df.alpha_y)
    warm = R[(R.n > 0) & (R.t_cam > R.t_cam.min() + 3)]
    gaps = np.diff(R.t_cam[sel].to_numpy())
    print(f"  alpha_x RMS vs direct measurement : "
          f"{np.rad2deg(np.sqrt(np.mean((R.alpha_x[sel]-mi[sel])**2))):.3f} deg")
    print(f"  alpha_y RMS vs direct measurement : "
          f"{np.rad2deg(np.sqrt(np.mean((R.alpha_y[sel]-mj[sel])**2))):.3f} deg")
    print(f"  mean normalised NIS               : "
          f"{np.mean(warm.nis/(2*warm.n)):.3f}  (target 1.0)")
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
    drone_ln, = ax.plot(E, N, U, color="#4c72b0", lw=1.8, label="drone")
    ref_ln = None
    if {"drone_px_ref", "drone_py_ref", "drone_pz_ref"} <= set(fl.columns):
        ref_ln, = ax.plot(fl.drone_px_ref, fl.drone_py_ref, fl.drone_pz_ref,
                          "k--", lw=1.2, label="reference")

    finite = np.flatnonzero(np.isfinite(pE))
    tethers = [ax.plot([E[k], pE[k]], [N[k], pN[k]], [U[k], pU[k]],
                       color="0.7", lw=0.5, alpha=0.7)[0]
               for k in finite[::25]]

    meas_ln, = ax.plot(pE, pN, pU, color="tab:orange", lw=1.0, alpha=0.85,
                       label="payload (camera measurement)")
    est_ln, = ax.plot(est_enu[:, 0], est_enu[:, 1], est_enu[:, 2],
                      color="tab:red", lw=1.8, label="payload (EKF estimate)")

    # the tether where the slider is; static until a slider is attached
    live_tether, = ax.plot([E[-1], est_enu[-1, 0]], [N[-1], est_enu[-1, 1]],
                           [U[-1], est_enu[-1, 2]], color="0.3", lw=1.6,
                           label="tether (at t)")

    ax.scatter([E[0]], [N[0]], [U[0]], c="k", s=40, label="start")
    head, = ax.plot([E[-1]], [N[-1]], [U[-1]], "s", ms=7, color="k",
                    mec="white", mew=0.8, label="drone at t")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_zlabel("Up [m]")
    ax.set_title("Drone and payload, ENU camera measurement vs EKF estimate")
    _set_equal_3d(ax,
                  np.concatenate([E, pE[finite], est_enu[:, 0]]),
                  np.concatenate([N, pN[finite], est_enu[:, 1]]),
                  np.concatenate([U, pU[finite], est_enu[:, 2]]))
    ax.legend(fontsize=8, loc="upper left")
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


def plot_timeseries(R, meas_df, offset, save=None):
    """EKF vs per-frame measurement, with occlusion shading and 2-sigma band."""
    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for k, (nm, lbl, sg) in enumerate(
            [("alpha_x", r"$\alpha_x$", "sigma_alpha_x"),
             ("alpha_y", r"$\alpha_y$", "sigma_alpha_y")]):
        a = axs[k]
        gaps = R.t_cam[R.n == 0].to_numpy()
        for t0 in gaps:
            a.axvspan(t0 - 0.015, t0 + 0.015, color="0.88", lw=0)
        a.plot(meas_df.t_cam, np.rad2deg(meas_df[nm]), ".", ms=4,
               color="tab:orange", label="per-frame camera measurement")
        a.plot(R.t_cam, np.rad2deg(R[nm]), "-", lw=1.6, color="tab:red",
               label="EKF estimate")
        a.fill_between(R.t_cam,
                       np.rad2deg(R[nm] - 2*R[sg]),
                       np.rad2deg(R[nm] + 2*R[sg]),
                       color="tab:red", alpha=0.2,
                       label=r"EKF $2\sigma$")
        a.set_ylabel(f"{lbl} [deg]")
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="upper right")

    axs[2].plot(R.t_cam, R.n, "k.", ms=3)
    axs[2].set_ylabel("markers seen")
    axs[2].set_yticks([0, 1, 2, 3])
    axs[2].set_xlabel("camera time [s]")
    axs[2].grid(alpha=0.3)
    axs[0].set_title("Payload swing: EKF vs direct measurement "
                     f"(grey = no marker detected, offset {offset:+.2f} s)")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=115)
        print("wrote", save)
    return fig


def animate_camera(records, save, fps=25, trail=40):
    """Line of sight in the camera frame: the (b_x, b_y) components of the unit
    bearings the filter is fed, against the ones it predicts.

    No intrinsics are involved, so this is not the image plane; it is the same
    view undistorted and scaled by the focal length, with the optical axis at
    the origin. The axes are still drawn image-style, +y downward.
    """
    fig, ax = plt.subplots(figsize=(8.2, 6.0))
    ax.set_aspect("equal")
    ax.set_xlabel(r"$b_x$  (camera frame, unit line of sight)")
    ax.set_ylabel(r"$b_y$")
    ax.grid(alpha=0.2)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.axvline(0, color="0.6", lw=0.8)

    # one set of limits for the whole run, so nothing jumps between frames
    pts = [b for r in records for b in r["meas"].values()]
    pts += [r["pred_c"] for r in records if r["pred_c"] is not None]
    pts = np.array([p[:2] for p in pts])
    pad = 0.05*max(np.ptp(pts, axis=0).max(), 1e-3)
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].max() + pad, pts[:, 1].min() - pad)   # +y downward

    meas_pts = {mid: ax.plot([], [], "o", ms=9, color=ID_COLOR[mid],
                             label=f"measured {ID_NAME[mid]}")[0] for mid in IDS}
    pred_pts = {mid: ax.plot([], [], "x", ms=10, mew=2.0,
                             color=ID_COLOR[mid], alpha=0.9)[0] for mid in IDS}
    center_pt, = ax.plot([], [], "*", ms=16, color="tab:red",
                         label="EKF payload center")
    trail_ln, = ax.plot([], [], "-", lw=1.0, color="tab:red", alpha=0.5)
    ell = Ellipse((0, 0), 1, 1, fc="tab:red", alpha=0.18, ec="tab:red", lw=1.0)
    ax.add_patch(ell)
    ax.plot([], [], "kx", ms=8, mew=2, label="EKF predicted marker")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    txt = ax.text(0.02, 0.02, "", transform=ax.transAxes, fontsize=9,
                  va="bottom", family="monospace",
                  bbox=dict(fc="white", alpha=0.8, ec="0.8"))
    hist = []

    def _at(artist, b):
        """Put a single-point artist on `b`, or clear it when there is none."""
        if b is None:
            artist.set_data([], [])
        else:
            artist.set_data([b[0]], [b[1]])

    def update(i):
        r = records[i]
        for mid in IDS:
            # measured dots follow the detections; the predicted x always shows
            # where the filter thinks each marker is, detected or not
            _at(meas_pts[mid], r["meas"].get(mid))
            _at(pred_pts[mid], r["pred"][mid])

        c = r["pred_c"]
        _at(center_pt, c)
        ell.set_visible(c is not None)
        if c is not None:
            hist.append(c)
            del hist[:-trail]
            hh = np.array(hist)
            trail_ln.set_data(hh[:, 0], hh[:, 1])
            S = r["S_c"]
            if S is not None:
                w, V = np.linalg.eigh(S)
                w = np.maximum(w, 1e-12)
                ell.set_center((c[0], c[1]))
                ell.width, ell.height = 2*np.sqrt(w[1]), 2*np.sqrt(w[0])
                ell.angle = math.degrees(math.atan2(V[1, 1], V[0, 1]))

        resid = ""
        if r["meas"]:
            # angle between two unit vectors, small enough that the norm will do
            e = [np.linalg.norm(r["meas"][m][:2] - r["pred"][m])
                 for m in r["meas"] if r["pred"][m] is not None]
            if e:
                resid = f"  resid {math.degrees(np.mean(e)):5.2f} deg"
        txt.set_text(f"t = {r['t_cam']:6.2f} s   markers {r['n']}{resid}\n"
                     f"alpha = ({math.degrees(r['alpha_x']):+6.2f}, "
                     f"{math.degrees(r['alpha_y']):+6.2f}) deg   "
                     f"1sig {math.degrees(r['sigma_alpha_x']):.2f} deg")
        return list(meas_pts.values()) + list(pred_pts.values()) + \
            [center_pt, trail_ln, ell, txt]

    ax.set_title(
        "Camera-frame bearings — measured markers vs EKF prediction")
    anim = animation.FuncAnimation(fig, update, frames=len(records),
                                   interval=1000/fps, blit=False)
    anim.save(str(save), writer=animation.FFMpegWriter(fps=fps, bitrate=2400))
    plt.close(fig)
    print("wrote", save)
    return save


# --------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("selector", nargs="?", default="latest.cam",
                    help="which session (see `uv run plot.py ls`)")
    ap.add_argument("--save", metavar="DIR", default=None,
                    help="write the figures and video here")
    ap.add_argument("--no-video", action="store_true",
                    help="skip the camera animation, which is slow")
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
    plot_timeseries(r["R"], r["meas_df"], r["offset"],
                    save=out and out / f"{session.id}_ekf_timeseries.png")

    if not args.no_video:
        mp4 = ((out / f"{session.id}_camera_view.mp4") if out else
               Path(tempfile.mkdtemp(prefix="tares_")) / "camera_view.mp4")
        animate_camera(r["records"], mp4)
        open_file(mp4)

    plt.show()   # interactive, so the 3-D view can be panned
