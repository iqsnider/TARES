"""
Run the payload EKF on a recorded flight and produce:
  1. a static 3D ENU plot of drone / measured payload / EKF-estimated payload
  2. a time-series comparison of the EKF against the per-frame camera
     measurement -- swing angles and payload yaw -- with occlusion shading
     and the 2-sigma band
  3. with --cam, the camera recording written back out as an .mp4 with the
     estimated payload position drawn on every frame

Both logs stamp every row with the same wall clock, so the offset between the
two comes off the files themselves (see `catalog.Session.pose_offset`). A run
recorded before poses.csv carried wall_time has no offset and cannot be run
through this.
"""
import colorsys
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import Prm.config as config
import sim.estimation.calculate_payload_position as payload
import sim.estimation.ekf as ekfm
import sim.estimation.pre_process as pp
import sim.transformations as tf
from sim.plotting import TimeSlider, configure_plot_style, C_REF, C_PAYLOAD
import catalog
import drone_plot
from run_on_log import build_inputs, group_frames, make_ekf, run_full

configure_plot_style()   # shared serif / Computer Modern theme

C_EST = "#009E73"
CALIB_FILE = (Path(__file__).resolve().parents[1] /
              "src/payload_tracking/camera_calibration/calibration.json")

BGR_EST = (60, 255, 80)
BGR_EST_REG = (255, 255, 0)  # vivid cyan, --compare's as-flown estimate
BGR_REF = (60, 60, 255)
BGR_ERR = (255, 60, 0)
BGR_AXIS = (245, 245, 245)
BGR_PLUMB = (255, 60, 200)
BGR_HOLD = (0, 200, 255)

# A point the lens never saw is an extrapolation of the distortion polynomial,
# which folds far-off-axis points back into the picture. The calibration only
# reaches tan ~0.83 at the frame corner, so the reference is cut a little
# beyond that: enough for the line to leave the frame, not enough to come back.
MAX_TAN = 1.2
MIN_DEPTH_M = 0.1

# How far ahead the velocity arrow reaches: one second of travel at the
# current swing velocity, which is a length rather than a gain that has to be
# looked up to be read.
LEAD_S = 1.0


def direct_measurement(frames, inp, offset, geom=None):
    """
    Per-frame swing angles straight from the PnP chain, no filtering.
    This is the orange scatter the EKF gets compared against. It is not
    truth -- it carries the same attitude error plus PnP depth noise.

    The same pre-processing the filter is fed, so the two are exactly
    comparable.
    """
    geom = pp.DEFAULT_GEOMETRY if geom is None else geom
    t_fl = inp["t"]
    rows = []
    for t_cam, g, det in frames:
        t = t_cam - offset
        if t < t_fl[0] or t > t_fl[-1]:
            continue
        phi = np.interp(t, t_fl, inp["phi"])
        theta = np.interp(t, t_fl, inp["theta"])
        psi = np.interp(t, t_fl, inp["psi"])
        S = tf.T_ENU_from_NED()
        m = pp.swing_angles(g, S @ tf.T_IB(phi, theta, psi) @ S, geom=geom)
        if m is None:
            continue
        rows.append((t_cam, *m, len(det)))
    return pd.DataFrame(rows, columns=["t_cam", "alpha_x", "alpha_y",
                                       "psi_p", "n"])


def estimated_payload_enu(records, fl, L_m):
    """p_payload = p_drone + L_m * q_I, drone position interpolated.

    Returns (t_flight, ENU) so the caller can put the estimate on a timeline.
    Pass the session's own config_snapshot.json TETHER_LEN as `L_m`.
    """
    t = fl.cur_time.to_numpy()
    t_flight = np.array([r["t_flight"] for r in records])
    E = np.interp(t_flight, t, fl.drone_px_meas.to_numpy())
    N = np.interp(t_flight, t, fl.drone_py_meas.to_numpy())
    U = np.interp(t_flight, t, fl.drone_pz_meas.to_numpy())
    q = np.array([r["q_I"] for r in records])
    return t_flight, np.c_[E, N, U] + L_m * q


# filter tuning a session snapshots, keyed by the EKF's own argument name
EKF_TUNING = {"q_xy": "EKF_Q_XY", "q_yaw": "EKF_Q_YAW",
              "sigma_xy": "EKF_SIGMA_XY", "sigma_yaw": "EKF_SIGMA_YAW",
              "sigma_alpha_0": "EKF_SIGMA_ALPHA_0",
              "sigma_rate_0": "EKF_SIGMA_RATE_0",
              "sigma_psi_p_0": "EKF_SIGMA_PSI_P_0",
              "zeta": "EKF_ZETA"}


def ekf_tuning(cfg):
    """A session's own filter tuning, so a replay is the filter that was flown.

    Runs recorded before the tuning was snapshotted never wrote it down, and
    those fall back to the airframe they flew rather than to whatever
    Prm/config.py is set to today -- otherwise replaying an old flight while
    the bench rig is selected runs it on the bench's numbers.
    """
    tuning = {arg: cfg[key] for arg, key in EKF_TUNING.items() if key in cfg}
    if not tuning:
        tuning = config.ekf_tuning_for(cfg["AIRFRAME"])

    return tuning


def hold_windows(fl):
    """Stretches of flight time where the run deliberately cut the camera.

    The bench test stamps sent_mode HOLD on those ticks, since nothing else is
    using that column when no setpoints go out. Empty for a real flight, whose
    gaps are occlusions rather than choices.
    """
    mode = fl.sent_mode.astype(str).to_numpy()
    t = fl.cur_time.to_numpy(float)
    held = mode == "HOLD"
    if not held.any():
        return []

    edges = np.flatnonzero(np.diff(held.astype(int)))
    bounds = np.concatenate([[0], edges + 1, [len(held)]])
    windows = [(t[a], t[b-1]) for a, b in zip(bounds[:-1], bounds[1:])
               if held[a]]

    return windows


def held_at(windows):
    """A predicate over flight time for `run_full`, or None if nothing held."""
    if not windows:
        return None

    def hold(t):
        return any(lo <= t <= hi for lo, hi in windows)

    return hold


def analyze(session, verbose=True):
    """Run the payload EKF over a session. Returns what the plots need.

    Geometry comes off session.config, not live Prm/config.py, so a session
    recorded before it had a config_snapshot.json raises. The 3-D track and
    the timeseries lines/sigma bands each prefer the onboard-logged estimate
    over a post-hoc reconstruction when the log has what they need
    (from_log, from_log_cov say which happened). The diagnostic prints
    (RMS vs. measurement, gaps) are always post-hoc: they need a per-frame
    detection count that was never logged onboard.
    """
    if not session.has_camera:
        raise SystemExit(f"session {session.id} has no camera data")
    if session.tracker == "color":
        raise SystemExit(
            f"session {session.id} was tracked with a color ring, and the "
            f"post-hoc filter reads marker poses. Its circles.csv holds the "
            f"ring center in x,y,z, so a color replay is possible but is not "
            f"written yet. Drop --post to draw the states the run logged.")

    geom = pp.Geometry.from_snapshot(session.config)
    offset = session.pose_offset
    inp = build_inputs(session.fl)
    frames = group_frames(session.poses)

    # L as well as geom: the pendulum frequency is the session's, not whatever
    # airframe Prm/config.py happens to be set to today
    windows = hold_windows(session.fl)
    records = run_full(frames, inp, offset, geom=geom,
                       hold=held_at(windows),
                       L=session.config["TETHER_LEN"],
                       source=ekfm.SOURCE_ARUCO,
                       **ekf_tuning(session.config))
    meas_df = direct_measurement(frames, inp, offset, geom=geom)
    if verbose:
        print(f"pose clock -> flight clock: {offset:+.3f} s")
        print(f"{len(records)} filter steps, "
              f"{sum(r['n'] > 0 for r in records)} with a measurement")
        for lo, hi in windows:
            print(f"camera held from {lo:.1f} to {hi:.1f} s, as the run did")
        summarise(records, meas_df)

    from_log = catalog.has_logged_states(session.fl)
    from_log_cov = from_log and catalog.has_logged_covariance(session.fl)
    track_records = logged_records(session) if from_log else records
    R = _records_frame(track_records if from_log_cov else records)
    if verbose:
        print("3-D track: " + ("onboard-logged estimate" if from_log
                               else "post-hoc reconstruction (no onboard log)"))
        print("timeseries (alpha/psi_p/sigma): " +
              ("onboard-logged" if from_log_cov else "post-hoc reconstruction"))

    est_t, est = estimated_payload_enu(track_records, session.fl,
                                       session.config["TETHER_LEN"])
    pdf = payload.get_payload_ENU_from_data(
        session.pose, session.fl, time_offset=offset, geom=geom,
        control_freq=session.config.get("CONTROL_FREQUENCY"))
    return dict(records=records, meas_df=meas_df, R=R, est_t=est_t, est=est,
                pdf=pdf, offset=offset, from_log=from_log,
                from_log_cov=from_log_cov)


_R_COLS = ("t_cam", "n", "alpha_x", "alpha_y", "psi_p",
          "sigma_alpha_x", "sigma_alpha_y", "sigma_psi_p")


def _records_frame(records):
    """
    The columns a timeseries plot needs, off records shaped like
    `run_full`'s or `logged_records`'s output.
    """
    df = pd.DataFrame([{k: v for k, v in r.items() if k in _R_COLS}
                       for r in records])
    return df


def summarise(records, meas_df):
    """
    Print the numbers worth checking after every run.

    Always over the post-hoc `records`, never `logged_records`: the RMS and
    gap numbers need a real per-frame detection count, never logged onboard.
    """
    R = _records_frame(records)
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
REF_COLS = {"payload": ("payload_px_ref", "payload_py_ref", "payload_pz_ref"),
            "drone": ("drone_px_ref", "drone_py_ref", "drone_pz_ref")}


def flown_reference(fl):
    """The reference this flight was tracking: ('payload'|'drone', Nx3 ENU).

    The log holds both sets of columns side by side and the controller fills
    in only the one it was following, so the pair that carries data says what
    the run was for. Drawing the payload reference where there is one is the
    whole comparison: the drone flies wherever it must, and what is supposed
    to land on the dashed line is the payload, not the aircraft. None when
    the run tracked neither -- a hand-flown or hover log.
    """
    which = catalog.flown_reference(fl)
    if which not in REF_COLS or not set(REF_COLS[which]) <= set(fl.columns):
        return None
    return which, fl[list(REF_COLS[which])].to_numpy(float)


def _set_equal_3d(ax, X, Y, Z):
    xmin, xmax = np.nanmin(X), np.nanmax(X)
    ymin, ymax = np.nanmin(Y), np.nanmax(Y)
    zmin, zmax = np.nanmin(Z), np.nanmax(Z)
    r = max(xmax-xmin, ymax-ymin, zmax-zmin) / 2 or 1.0
    cx, cy, cz = (xmax+xmin)/2, (ymax+ymin)/2, (zmax+zmin)/2
    ax.set_xlim(cx-r, cx+r)
    ax.set_ylim(cy-r, cy+r)
    ax.set_zlim(cz-r, cz+r)
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def _mode_lines(ax, fl, E, N, U, lw=2.0):
    """Drone track split into flight-mode runs, colored like drone_plot's."""
    lines, seen = [], set()
    for i, j, mode in drone_plot._mode_runs(fl):
        color = drone_plot.MODE_SHADING.get(mode, ("0.7", 0))[0]
        sl = slice(i, min(j + 2, len(fl)))
        lbl = f"Drone ({mode})" if mode not in seen else None
        ln, = ax.plot(E[sl], N[sl], U[sl], color=color, lw=lw, label=lbl)
        lines.append(ln)
        seen.add(mode)
    return lines


def plot_3d(fl, payload_df, est_t, est_enu, save=None, slider=True,
           from_log=None):
    """
    Drone/payload/EKF paths in ENU.

    `est_t` is the flight time of each row of `est_enu`. With `slider` the
    on-screen figure gets a time slider that plays the run back; the saved PNG
    is written first, so the file always holds the whole flight. `from_log`
    (see `analyze`) sets the legend to say whether `est_enu` is onboard-logged
    or reconstructed.
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
    drone_lns = _mode_lines(ax, fl, E, N, U)

    # whichever of the two references this run was flying, so a payload-
    # tracking run is not silently drawn against an empty drone reference
    ref = flown_reference(fl)
    ref_ln, ref_xyz = None, np.empty((0, 3))
    if ref is not None:
        which, ref_xyz = ref
        # over the tracks rather than under them: a payload reference runs
        # right where the payload does, which is the point, and a 1.6pt line
        # drawn first vanishes beneath the 2pt one it is being compared with
        ref_ln, = ax.plot(ref_xyz[:, 0], ref_xyz[:, 1], ref_xyz[:, 2],
                          color=C_REF, lw=1.6, linestyle=(0, (5, 4)),
                          zorder=5, label=f"{which.capitalize()} reference")

    finite = np.flatnonzero(np.isfinite(pE))
    tethers = [ax.plot([E[k], pE[k]], [N[k], pN[k]], [U[k], pU[k]],
                       color="#5F6368", lw=0.9, alpha=0.35,
                       label="Tether (snapshots)" if j == 0 else None)[0]
               for j, k in enumerate(finite[::25])]

    meas_ln, = ax.plot(pE, pN, pU, color=C_PAYLOAD, lw=1.4, alpha=0.85,
                       label="Payload (camera measurement)")
    est_label = {True: "Payload (EKF estimate, onboard)",
                False: "Payload (EKF estimate, reconstructed)",
                None: "Payload (EKF estimate)"}[from_log]
    est_ln, = ax.plot(est_enu[:, 0], est_enu[:, 1], est_enu[:, 2],
                      color=C_EST, lw=2.0, label=est_label)

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
    # the reference belongs in the box too: a payload reference sits a tether
    # below the drone, so leaving it out would crop it off the bottom
    ref_ok = ref_xyz[np.isfinite(ref_xyz).all(axis=1)]
    _set_equal_3d(ax,
                  np.concatenate([E, pE[finite], est_enu[:, 0], ref_ok[:, 0]]),
                  np.concatenate([N, pN[finite], est_enu[:, 1], ref_ok[:, 1]]),
                  np.concatenate([U, pU[finite], est_enu[:, 2], ref_ok[:, 2]]))

    # one legend entry per mode, not one per segment
    labeled_drone_lns = [ln for ln in drone_lns if not ln.get_label().startswith("_")]

    # same running order as the simulation figure, which draws its artists in
    # a different sequence than this one does
    order = [ref_ln, *labeled_drone_lns, meas_ln, est_ln,
             tethers[0] if tethers else None, live_tether, start_pt, head]
    ax.legend(handles=[h for h in order if h is not None],
              fontsize=13.5, loc="upper left")
    fig.tight_layout()
    if save:
        fig.savefig(save, dpi=120)
        print("wrote", save)

    if slider:
        s = TimeSlider(fig, t, label="flight time [s]")
        for (i, j, mode), ln in zip(drone_plot._mode_runs(fl), drone_lns):
            sl = slice(i, min(j + 2, len(fl)))
            s.line(ln, np.c_[E, N, U][sl], t[sl])
        s.line(meas_ln, np.c_[pE, pN, pU], payload_df.cur_time.to_numpy())
        s.line(est_ln, est_enu, est_t)
        s.marker(head, np.c_[E, N, U])
        # to the EKF estimate, the one payload track with no detection gaps
        s.span(live_tether, np.c_[E, N, U], est_enu, t_b=est_t)
        s.group(tethers, t[finite[::25]])
        if ref_ln is not None:
            s.line(ref_ln, ref_xyz)
    return fig


def _break_wraps(deg, jump=180.0):
    """NaN the sample after a +-180 wrap, so the line does not draw a vertical
    stripe across the panel. Only psi_p wraps."""
    deg = np.asarray(deg, float).copy()
    deg[np.r_[False, np.abs(np.diff(deg)) > jump]] = np.nan
    return deg


def plot_timeseries(R, meas_df, save=None, from_log_cov=None):
    """
    EKF vs per-frame measurement, with the 2-sigma band.

    `from_log_cov` (see `analyze`) sets the legend to say whether `R` is
    onboard-logged or a post-hoc reconstruction.
    """
    est_label = {True: "EKF estimate (onboard)",
                False: "EKF estimate (reconstructed)",
                None: "EKF estimate"}[from_log_cov]
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
               label=est_label)
        a.fill_between(R.t_cam, y - 2*sig, y + 2*sig,
                       color="tab:red", alpha=0.2,
                       label=r"EKF $2\sigma$")
        a.set_ylabel(f"{lbl} [deg]")
        a.grid(alpha=0.3)
        a.legend(fontsize=8, loc="upper right")

    # off meas_df, not R: R.n is a placeholder (0 for every row) when R comes
    # from the onboard log, which never recorded a per-frame detection count;
    # meas_df is always the fresh per-frame measurement, so its n is real
    axs[3].plot(meas_df.t_cam, meas_df.n, "k.", ms=3)
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
OVERLAY_SUFFIX = "_ekf_overlay.mp4"


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


def _label(frame, c, lines, color=BGR_EST, gap=14, scale=1.15, thick=3):
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
        cv2.putText(frame, text, at, font, scale, color, thick, cv2.LINE_AA)


def _hex_to_bgr(color):
    """Matplotlib color string (hex, or a grayscale level like "0.7") to BGR."""
    if color.startswith("#"):
        r, g, b = (int(color[i:i+2], 16) for i in (1, 3, 5))
        bgr = (b, g, r)
        return bgr
    v = round(float(color)*255)
    bgr = (v, v, v)
    return bgr


def _vivid(bgr):
    """Same hue as `bgr`, at full saturation and brightness.

    The plot colors (MODE_SHADING) are muted, meant for translucent shading;
    this pops on video while staying visibly the same color as the plots.
    """
    b, g, r = bgr
    h, _, _ = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    r, g, b = colorsys.hsv_to_rgb(h, 1, 1)
    vivid_bgr = (round(b*255), round(g*255), round(r*255))
    return vivid_bgr


def _mode_by_frame(session):
    """Flight mode at each camera frame, nearest sample in flight time."""
    frames = session.poses.groupby("frame").time_s.first()
    t_flight = frames.to_numpy(float) - session.pose_offset
    fl = session.fl
    t_fl = fl.cur_time.to_numpy()
    modes = fl.echoed_mode.astype(str).to_numpy()
    idx = np.searchsorted(t_fl, t_flight).clip(1, len(t_fl) - 1)
    idx -= (t_flight - t_fl[idx - 1]) < (t_fl[idx] - t_flight)
    by_frame = dict(zip(frames.index.astype(int), modes[idx]))
    return by_frame


def _crosshair(frame, K):
    """Lines through the camera center, for reading the velocity arrow against.

    The principal point from the calibration rather than the middle of the
    picture, though on this lens the two sit within about a pixel.
    """
    h, w = frame.shape[:2]
    cx, cy = round(K[0, 2]), round(K[1, 2])
    cv2.line(frame, (0, cy), (w, cy), BGR_AXIS, 2, cv2.LINE_AA)
    cv2.line(frame, (cx, 0), (cx, h), BGR_AXIS, 2, cv2.LINE_AA)


def _corner_text(frame, lines, corner, color=(255, 255, 255), scale=1, thick=2,
                 margin=12, dy=0):
    """Draw `lines` anchored to a fixed screen corner, outlined for legibility.

    `dy` pushes the block down, to sit under a line already drawn there.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    step = int(30*scale)
    widths = [cv2.getTextSize(s, font, scale, thick)[0][0] for s in lines]
    right = corner == "right"
    for k, (text, width) in enumerate(zip(lines, widths)):
        x = frame.shape[1] - margin - width if right else margin
        y = margin + dy + step*(k+1)
        cv2.putText(frame, text, (x, y), font, scale, (0, 0, 0), thick + 3,
                    cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), font, scale, color, thick, cv2.LINE_AA)


def _config_summary_lines(cfg):
    """Airframe/tether/mass/controller/filter readout for a session's config_snapshot.json."""
    def g(key, fmt="{:.2f}"):
        v = cfg.get(key)
        text = fmt.format(v) if v is not None else "n/a"
        return text

    def deg(key, fmt="{:.2f}"):
        v = cfg.get(key)
        text = fmt.format(np.degrees(v)) if v is not None else "n/a"
        return text
    lines = [
        f"airframe {cfg.get('AIRFRAME', 'n/a')}",
        f"controller {cfg.get('CONTROLLER', 'n/a')}",
        f"tether {g('TETHER_LEN')} m",
        f"drone mass {g('MASS_DRONE')} kg",
        f"payload mass {g('MASS_PAYLOAD')} kg",
        f"LQR w_xy={g('LQR_PAYLOAD_W_POS_XY', '{:.3f}')} "
        f"w_z={g('LQR_PAYLOAD_W_POS_Z', '{:.3f}')} "
        f"tune={g('LQR_PAYLOAD_TUNING_CONST', '{:.3f}')}",
        f"EKF sig_xy={deg('EKF_SIGMA_XY')} deg "
        f"sig_yaw={deg('EKF_SIGMA_YAW', '{:.0f}')} deg "
        f"q_xy={g('EKF_Q_XY', '{:.4f}')}",
    ]
    return lines


def detections_by_frame(session):
    """
    How many detections each camera frame carried, from the tracker's own CSV.

    A frame with nothing in it is still written, with the detection columns
    left empty, so counting the filled ones is what says whether the filter
    had anything to fold in.
    """
    col = "u_px" if session.tracker == "color" else "marker_id"
    poses = session.poses
    if col not in poses:
        return {}

    counts = poses.dropna(subset=[col]).groupby("frame").size()

    return {int(f): int(k) for f, k in counts.items()}


def logged_records(session):
    """
    The onboard EKF's own output, per camera frame, shaped like `run_full`.

    The states here are what the aircraft actually flew on, not a
    reconstruction. `catalog.has_logged_covariance` gates psi_p and `P`: a
    flight recorded before those were logged gets psi_p=0 and P=None instead
    of a fabricated number. `P` is only the alpha_x/alpha_y/psi_p block, not
    the full state covariance. `n` counts what the tracker saw on that frame,
    read back from its own CSV since the onboard log never recorded it.
    """
    fl = session.fl
    t = fl.cur_time.to_numpy()
    frames = session.poses.groupby("frame").time_s.first()
    seen = detections_by_frame(session)
    has_cov = catalog.has_logged_covariance(fl)
    S = tf.T_ENU_from_NED()

    t_cam = frames.to_numpy(float)
    t_flight = t_cam - session.pose_offset
    keep = (t_flight >= t[0]) & (t_flight <= t[-1])
    n = keep.sum()

    def at(col):
        return np.interp(t_flight[keep], t, fl[col].to_numpy(float))

    psi_p_col = at("payload_psi_p") if has_cov else np.zeros(n)
    xi = np.column_stack([at(c) for c in catalog.LOGGED_STATE_COLS] + [psi_p_col])
    phi, theta, psi = (at("drone_roll"), at("drone_pitch"), at("drone_yaw"))
    nan = float("nan")

    if has_cov:
        axx, ayy, axy, app = (at("payload_cov_axx"), at("payload_cov_ayy"),
                              at("payload_cov_axy"), at("payload_cov_psipsi"))
        sigma_ax, sigma_ay = np.sqrt(axx), np.sqrt(ayy)
        sigma_pp = np.sqrt(app)

        def P_at(i):
            P = np.zeros((ekfm.STATE_DIM, ekfm.STATE_DIM))
            P[ekfm.IX_ALPHA_X, ekfm.IX_ALPHA_X] = axx[i]
            P[ekfm.IX_ALPHA_Y, ekfm.IX_ALPHA_Y] = ayy[i]
            P[ekfm.IX_ALPHA_X, ekfm.IX_ALPHA_Y] = axy[i]
            P[ekfm.IX_ALPHA_Y, ekfm.IX_ALPHA_X] = axy[i]
            P[ekfm.IX_PSI_P, ekfm.IX_PSI_P] = app[i]
            return P
    else:
        sigma_ax = sigma_ay = sigma_pp = np.full(n, nan)
        P_at = lambda i: None

    return [dict(frame=int(f), t_cam=tc, t_flight=t_fl_i, n=seen.get(int(f), 0),
                 xi=xi[i], P=P_at(i),
                 T_IB=S @ tf.T_IB(phi[i], theta[i], psi[i]) @ S,
                 alpha_x=xi[i, ekfm.IX_ALPHA_X], alpha_y=xi[i, ekfm.IX_ALPHA_Y],
                 alpha_dot_x=xi[i, ekfm.IX_ALPHA_DOT_X],
                 alpha_dot_y=xi[i, ekfm.IX_ALPHA_DOT_Y],
                 psi_p=xi[i, ekfm.IX_PSI_P],
                 sigma_alpha_x=sigma_ax[i], sigma_alpha_y=sigma_ay[i],
                 sigma_psi_p=sigma_pp[i],
                 q_I=np.array([xi[i, ekfm.IX_ALPHA_X], xi[i, ekfm.IX_ALPHA_Y], -1]))
            for i, (f, tc, t_fl_i) in enumerate(zip(frames.index[keep],
                                                    t_cam[keep], t_flight[keep]))]


def project_enu(P_I, p_drone, T_IB, filt, K, D):
    """Pixel coordinates of ENU points `P_I` seen from the camera at one frame.

    The same chain `EKF.estimate_to_px_coords` walks, written for an arbitrary
    inertial point instead of a state estimate: into the body frame about the
    drone, across the camera lever arm, into the camera frame, through the
    lens. Rows that fall behind the camera or outside the calibrated cone come
    back NaN rather than somewhere wrong.
    """
    p_C = ((np.atleast_2d(P_I) - p_drone) @ T_IB - filt.t_BC_B) @ filt.T_CB.T
    ok = p_C[:, 2] > MIN_DEPTH_M
    ok &= np.hypot(p_C[:, 0], p_C[:, 1]) < MAX_TAN*np.abs(p_C[:, 2])

    uv = np.full((len(p_C), 2), np.nan)
    if ok.any():
        px, _ = cv2.projectPoints(p_C[ok].reshape(-1, 1, 3), np.zeros(3),
                                  np.zeros(3), K, D)
        uv[ok] = px.reshape(-1, 2)
    return uv


def _runs(uv):
    """The drawable stretches of a projected path: pixels, split at the gaps.

    A path that leaves the camera's view and comes back must not be joined
    across the part that was never seen, so it is drawn as several polylines.
    """
    idx = np.flatnonzero(np.isfinite(uv[:, 0]))
    if idx.size == 0:
        return []
    parts = np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
    return [np.clip(uv[p], -1e4, 1e4).round().astype(np.int32)
            for p in parts if p.size > 1]


def drone_states(fl, records):
    """(position, velocity) in ENU at each record's flight time, Nx3 each.

    The camera runs on its own clock and its own rate, so everything the
    overlay needs off the flight log is resampled onto the frames once, here,
    rather than a column at a time down in the drawing.
    """
    t = fl.cur_time.to_numpy()
    at = [r["t_flight"] for r in records]

    def cols(*names):
        return np.column_stack([np.interp(at, t, fl[c].to_numpy(float))
                                for c in names])

    return (cols("drone_px_meas", "drone_py_meas", "drone_pz_meas"),
            cols("drone_vx_meas", "drone_vy_meas", "drone_vz_meas"))


def body_rates(fl, records):
    """
    Body angular rate at each record's flight time, ENU ordered, Nx3 [rad/s].

    RAW_IMU logs the gyro in mrad/s about FRD, so it is scaled and swapped
    into the same body ENU the overlay's T_IB is built on.
    """
    t = fl.cur_time.to_numpy()
    at = [r["t_flight"] for r in records]

    w_frd = np.column_stack([np.interp(at, t, fl[c].to_numpy(float))/1000
                             for c in ("drone_gyrox_meas", "drone_gyroy_meas",
                                       "drone_gyroz_meas")])

    w_B = w_frd @ tf.T_ENU_from_NED().T

    return w_B


def payload_enu(r, p_drone, filt):
    """Where the filter puts the payload: a tether below the drone, leaned
    over by the swing angles. The small-angle model the controller flew on."""
    return p_drone + filt.L*np.array([r["xi"][ekfm.IX_ALPHA_X],
                                      r["xi"][ekfm.IX_ALPHA_Y], -1])


def velocity_tip_px(r, p_drone, w_B, filt, K, D, lead_s=LEAD_S):
    """Where the payload is headed, in pixels: the tip of the velocity arrow.

    The payload's velocity in the camera frame, which is the motion the
    picture actually shows. The camera rides the drone, so the drone's own
    travel drops out and the swing is left,

        v_rel_I = L*[alphadot_x, alphadot_y, 0]

    but the camera also turns with the drone, and a turning camera sweeps the
    payload across the frame with no swing at all. That is the transport term,
    taken off in the body frame where the rate is measured,

        v_seen = v_rel - w x d,   d = drone to payload

    On a long tether it is the larger half: at 7.5 m a 0.09 rad/s body rate
    already outruns a typical swing. A payload flying level with the drone,
    on a drone holding its attitude, sits still in frame and carries no arrow.
    The ENU velocity the controller is judged on is the payload velocity plot
    instead.

    The tip is the payload's own position carried `lead_s` along that velocity
    and projected like any other point, so the arrow reads as a length rather
    than an arbitrary gain: it spans one second of travel at the current
    speed, foreshortening and lens distortion included. That is a scale, not a
    prediction -- the tether is a 6.3 s pendulum, so a second of swing turns
    through a sixth of a cycle and the payload never reaches the tip.
    """
    T_IB = r["T_IB"]
    v_rel_I = filt.L*np.array([r["xi"][ekfm.IX_ALPHA_DOT_X],
                               r["xi"][ekfm.IX_ALPHA_DOT_Y], 0])

    # drone to payload, carried into the body frame the gyro measures in
    d_I = filt.L*np.array([r["xi"][ekfm.IX_ALPHA_X],
                           r["xi"][ekfm.IX_ALPHA_Y], -1])
    v_rot_I = T_IB @ np.cross(w_B, T_IB.T @ d_I)

    ahead = payload_enu(r, p_drone, filt) + lead_s*(v_rel_I - v_rot_I)

    tip = project_enu(ahead, p_drone, T_IB, filt, K, D)[0]

    return tip


def plumb_px(p_drone, T_IB, filt, K, D):
    """Straight down from the drone, one tether length, in pixels.

    Where the payload would hang with no swing at all, so the green dot's
    distance from it is the swing itself. Not the same point as the camera
    center: the camera sits off the drone's CG and the picture leans with the
    drone's attitude, while this hangs on gravity.
    """
    below = np.asarray(p_drone, float) + np.array([0, 0, -filt.L])

    uv = project_enu(below, p_drone, T_IB, filt, K, D)[0]

    return uv


def reference_pixels(session, records, filt, K, D, stride=5, truncate=False):
    """{frame: (path stretches, the point being tracked right now, error [m])}.

    The reference is a track through the world, so it is drawn as one, with
    the sample the controller was chasing at that instant marked on it. Both
    move about the picture as the drone flies, and where the payload sits
    against them is the tracking error the camera can actually show. Empty
    when the run flew no reference.

    With `truncate`, only the reference up to the current frame is drawn --
    for a stick test the "future" of the reference does not exist yet, it is
    wherever the pilot moves the stick next, so drawing the whole track like
    a planned flight would show a path that was never really there.

    The error is the straight-line distance in ENU from that sample to where
    the filter puts the payload, measured in the world rather than off the
    picture: two points the same distance apart look closer together the
    further from the lens axis they sit, so pixels would not be a length.
    """
    ref = flown_reference(session.fl)
    if ref is None:
        return {}
    _, ref_xyz = ref

    t = session.fl.cur_time.to_numpy()
    p_drone, _ = drone_states(session.fl, records)
    # the whole path is redrawn every frame, so it is thinned first; the log
    # runs at 50 Hz and the reference crawls, well under a pixel per sample
    path_full = ref_xyz[::stride]

    out = {}
    for k, r in enumerate(records):
        here = np.array([np.interp(r["t_flight"], t, ref_xyz[:, c])
                         for c in range(3)])
        if truncate:
            cutoff = np.searchsorted(t, r["t_flight"], side="right")
            path = ref_xyz[:cutoff][::stride]
        else:
            path = path_full
        uv = project_enu(np.vstack([path, here]), p_drone[k], r["T_IB"],
                         filt, K, D)
        gap = np.linalg.norm(payload_enu(r, p_drone[k], filt) - here)
        out[r["frame"]] = (_runs(uv[:-1]), uv[-1], float(gap))
    return out


def overlay_video(session, records, save=None):
    """Redraw the recording with the EKF payload estimate on each frame.

    A green dot marks where the filter believes the payload is, projected into
    the camera by `EKF.estimate_to_px_coords`. A green arrow off the dot is the
    payload's velocity in the camera frame, one second of travel long, drawn
    where that second would take it, and gray lines through the camera center
    give it something to be read against. A purple dot hangs one tether length
    straight below the drone, so the swing is the gap between purple and green.
    The reference the run was flying is drawn in
    red -- the whole track as a line, the sample being chased now as a dot --
    so the estimate can be read against what it was supposed to be. A blue
    line closes the gap between the two, labelled with how far apart they are
    in metres. Frames the filter never reached are copied through untouched.

    The geometry used to project the estimate into pixels comes from the
    session's own config_snapshot.json (session.config), not the live
    Prm/config.py.

    The current flight mode is drawn top-left, colored like the mode-shaded
    plots (drone_plot.MODE_SHADING), and under it whether that frame gave the
    filter a measurement at all. The session's tether length, drone/payload
    mass, and payload LQR costs are drawn top-right.
    """
    geom = pp.Geometry.from_snapshot(session.config)
    mode_by_frame = _mode_by_frame(session)
    config_lines = _config_summary_lines(session.config)
    src = find_recording(session)
    if src is None:
        raise SystemExit(f"session {session.id} has no recording next to "
                         f"{session.pose.name}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # the source video's own header fps is only the rate that was requested,
    # not what the camera actually achieved; poses.csv time_s is the ground
    # truth, and this pass already reads every frame regardless, so using it
    # here costs nothing extra
    t_pose = session.poses.groupby("frame").time_s.first().to_numpy()
    fps = (len(t_pose) - 1) / (t_pose[-1] - t_pose[0]) if len(t_pose) > 1 else 30

    out = Path(save) if save else src.with_name(session.id + OVERLAY_SUFFIX)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    K, D = _intrinsics()
    filt = make_ekf(0, 0, 0, 0, 0, 0, geom=geom,
                    source=ekfm.SOURCE_ARUCO,
                    L=session.config.get("TETHER_LEN"))
    p_drone, _ = drone_states(session.fl, records)
    w_B = body_rates(session.fl, records)
    meas_by_frame = {r["frame"]: r["n"] for r in records}
    drawn = {}
    for k, r in enumerate(records):
        # P only ever fed the uncertainty ellipse, so zeros do here
        P = np.zeros((ekfm.STATE_DIM, ekfm.STATE_DIM))
        center, _, _ = filt.estimate_to_px_coords(r["xi"], P, r["T_IB"], K, D)
        # a payload swung out of the camera's half-space projects nowhere
        if np.isfinite(center).all():
            drawn[r["frame"]] = (center,
                                 velocity_tip_px(r, p_drone[k], w_B[k], filt, K, D),
                                 plumb_px(p_drone[k], r["T_IB"], filt, K, D))
    ref = reference_pixels(session, records, filt, K, D,
                           truncate="stick" in session.label.lower())

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        _crosshair(frame, K)

        mode = mode_by_frame.get(i)
        if mode is not None:
            mode_color = _vivid(_hex_to_bgr(
                drone_plot.MODE_SHADING.get(mode, ("0.7", 0))[0]))
            _corner_text(frame, [mode], "left", color=mode_color, scale=2, thick=4)

        # whether this frame fed the filter anything, so a blackout is visible
        # the moment it starts rather than inferred from the estimate drifting
        n = meas_by_frame.get(i)
        if n is not None:
            _corner_text(frame, ["MEASUREMENT" if n else "NO MEASUREMENT"],
                         "left", color=BGR_EST if n else BGR_HOLD,
                         scale=1.4, thick=3, dy=64 if mode is not None else 0)
        _corner_text(frame, config_lines, "right")

        # under the estimate: where the payload is beats where it should be
        line = ref.get(i)
        at_ref = None
        if line is not None:
            runs, here, err_m = line
            cv2.polylines(frame, runs, False, BGR_REF, 2, cv2.LINE_AA)
            if np.isfinite(here).all():
                at_ref = tuple(np.clip(here, -1e4, 1e4).round().astype(int))
                cv2.circle(frame, at_ref, 10, BGR_REF, -1, cv2.LINE_AA)
                cv2.circle(frame, at_ref, 10, (255, 255, 255), 2, cv2.LINE_AA)

        hit = drawn.get(i)
        if hit is not None:
            center, tip, plumb = hit
            c = tuple(np.round(center).astype(int))

            # under the estimate: the swing is the gap between the two
            if np.isfinite(plumb).all():
                at_plumb = tuple(np.clip(plumb, -1e4, 1e4).round().astype(int))
                cv2.circle(frame, at_plumb, 6, BGR_PLUMB, -1, cv2.LINE_AA)
                cv2.circle(frame, at_plumb, 6, (255, 255, 255), 2, cv2.LINE_AA)

            # the gap the run is being judged on, drawn where it happens and
            # labelled with the length it actually is
            if at_ref is not None:
                cv2.line(frame, at_ref, c, BGR_ERR, 2, cv2.LINE_AA)
                mid = ((at_ref[0] + c[0])//2, (at_ref[1] + c[1])//2)
                _label(frame, mid, (f"{err_m:.2f} m",), color=BGR_ERR)

            if np.isfinite(tip).all():
                cv2.arrowedLine(frame, c,
                                tuple(np.clip(tip, -1e4, 1e4).round().astype(int)),
                                BGR_EST, 2, cv2.LINE_AA, tipLength=0.2)

            cv2.circle(frame, c, 6, BGR_EST, -1)
            cv2.circle(frame, c, 6, (255, 255, 255), 2)
        writer.write(frame)
        i += 1

    cap.release()
    writer.release()
    print(f"wrote {out}  ({len(drawn)} of {i} frames carry an estimate)")
    return out


def overlay_compare_video(session, records_reg, records_post, save=None):
    """Redraw the recording with both the as-flown and post-hoc estimates on it.

    Same picture as `overlay_video` -- reference track, plumb bob, corner text
    -- but the estimate dot and velocity arrow are drawn twice: vivid cyan
    for `records_reg` (what the aircraft actually flew on, or a live run if it
    never logged states) and green for `records_post` (the filter re-run
    offline on the session's own config_snapshot.json). The plumb bob is drawn
    once, from whichever of the two has an estimate that frame, since it only
    depends on drone attitude and does not move between the two runs.
    """
    geom = pp.Geometry.from_snapshot(session.config)
    mode_by_frame = _mode_by_frame(session)
    config_lines = _config_summary_lines(session.config)
    src = find_recording(session)
    if src is None:
        raise SystemExit(f"session {session.id} has no recording next to "
                         f"{session.pose.name}")

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    t_pose = session.poses.groupby("frame").time_s.first().to_numpy()
    fps = (len(t_pose) - 1) / (t_pose[-1] - t_pose[0]) if len(t_pose) > 1 else 30

    out = Path(save) if save else src.with_name(session.id + "_compare" + OVERLAY_SUFFIX)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    K, D = _intrinsics()
    filt = make_ekf(0, 0, 0, 0, 0, 0, geom=geom,
                    source=ekfm.SOURCE_ARUCO,
                    L=session.config.get("TETHER_LEN"))

    def project(records):
        p_drone, _ = drone_states(session.fl, records)
        w_B = body_rates(session.fl, records)
        drawn = {}
        for k, r in enumerate(records):
            P = np.zeros((ekfm.STATE_DIM, ekfm.STATE_DIM))
            center, _, _ = filt.estimate_to_px_coords(r["xi"], P, r["T_IB"], K, D)
            if np.isfinite(center).all():
                drawn[r["frame"]] = (center,
                                     velocity_tip_px(r, p_drone[k], w_B[k], filt, K, D),
                                     plumb_px(p_drone[k], r["T_IB"], filt, K, D))
        return drawn

    drawn_reg = project(records_reg)
    drawn_post = project(records_post)
    meas_by_frame = {r["frame"]: r["n"] for r in records_reg}
    ref = reference_pixels(session, records_reg, filt, K, D,
                           truncate="stick" in session.label.lower())

    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        _crosshair(frame, K)

        mode = mode_by_frame.get(i)
        if mode is not None:
            mode_color = _vivid(_hex_to_bgr(
                drone_plot.MODE_SHADING.get(mode, ("0.7", 0))[0]))
            _corner_text(frame, [mode], "left", color=mode_color, scale=2, thick=4)

        n = meas_by_frame.get(i)
        if n is not None:
            _corner_text(frame, ["MEASUREMENT" if n else "NO MEASUREMENT"],
                         "left", color=BGR_EST if n else BGR_HOLD,
                         scale=1.4, thick=3, dy=64 if mode is not None else 0)
        _corner_text(frame, config_lines, "right")
        # fixed offset, clear of the mode/measurement lines above regardless
        # of whether either was drawn this frame
        _corner_text(frame, ["as-flown"], "left", color=BGR_EST_REG, dy=150)
        _corner_text(frame, ["post-hoc"], "left", color=BGR_EST, dy=180)

        line = ref.get(i)
        if line is not None:
            runs, here, _ = line
            cv2.polylines(frame, runs, False, BGR_REF, 2, cv2.LINE_AA)
            if np.isfinite(here).all():
                at_ref = tuple(np.clip(here, -1e4, 1e4).round().astype(int))
                cv2.circle(frame, at_ref, 10, BGR_REF, -1, cv2.LINE_AA)
                cv2.circle(frame, at_ref, 10, (255, 255, 255), 2, cv2.LINE_AA)

        hit_reg = drawn_reg.get(i)
        hit_post = drawn_post.get(i)

        plumb_hit = hit_reg or hit_post
        if plumb_hit is not None:
            _, _, plumb = plumb_hit
            if np.isfinite(plumb).all():
                at_plumb = tuple(np.clip(plumb, -1e4, 1e4).round().astype(int))
                cv2.circle(frame, at_plumb, 6, BGR_PLUMB, -1, cv2.LINE_AA)
                cv2.circle(frame, at_plumb, 6, (255, 255, 255), 2, cv2.LINE_AA)

        for hit, color in ((hit_reg, BGR_EST_REG), (hit_post, BGR_EST)):
            if hit is None:
                continue
            center, tip, _ = hit
            c = tuple(np.round(center).astype(int))

            if np.isfinite(tip).all():
                cv2.arrowedLine(frame, c,
                                tuple(np.clip(tip, -1e4, 1e4).round().astype(int)),
                                color, 2, cv2.LINE_AA, tipLength=0.2)

            cv2.circle(frame, c, 6, color, -1)
            cv2.circle(frame, c, 6, (255, 255, 255), 2)

        writer.write(frame)
        i += 1

    cap.release()
    writer.release()
    print(f"wrote {out}  ({len(drawn_reg)} as-flown / {len(drawn_post)} "
          f"post-hoc frames carry an estimate)")
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

    r = analyze(session)

    out = Path(args.save).expanduser() if args.save else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    plot_3d(session.fl, r["pdf"], r["est_t"], r["est"],
            save=out and out / f"{session.id}_ekf_3d.png",
            from_log=r["from_log"])
    plot_timeseries(r["R"], r["meas_df"],
                    save=out and out / f"{session.id}_ekf_timeseries.png",
                    from_log_cov=r["from_log_cov"])

    plt.show()   # interactive, so the 3-D view can be panned
