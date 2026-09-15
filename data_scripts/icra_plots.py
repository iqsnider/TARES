"""
The 11 Sep figures: payload tracks over the square, the position error and
the swing after each arrival, and the RMS numbers, for two payload runs and
both baselines, plus the piloting aid track and its reference velocity.

    uv run data_scripts/icra_plots.py            print and show
    uv run data_scripts/icra_plots.py figs/      print and write there

Tracks are the payload states the aircraft logged. The plotted line is
broken where the flight controller reset its position estimate or the onboard
filter took a false ring detection, so neither draws as a swing.
"""

from paper_plot import _frame, C_INK, LEGEND_SIZE
from icra import C_PAYLOAD_CTRL, C_ARDUPILOT
from run_on_log import build_inputs, group_circles, run_full
from drone_plot import payload_enu
import sim.estimation.pre_process as pp
import sim.estimation.ekf as ekfm
import ekf_plots
import catalog
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42


RUNS = [("09112026.100158", "payload"),
        ("09112026.102857", "payload"),
        ("09112026.100414", "ardupilot"),
        ("09112026.102556", "ardupilot")]

NAMES = {"ardupilot": "ArduPilot Baseline",
         "payload": "Payload Control"}
COLORS = {"ardupilot": C_ARDUPILOT, "payload": C_PAYLOAD_CTRL}

# the piloting aid, flown after the squares. Only the stretch under payload
# control is plotted; the pilot flew the rest on ardupilot's own loiter
STICK_RUN = "09112026.111933"
STICK_RANGE_TOL = 0.2   # of the tether: the range a real ring detection sits at

# ENU components, kept apart by colour rather than by line style
C_EAST = "#E69F00"
C_NORTH = "#009E73"
C_UP = "#8856A7"

RESET_M = 0.3      # drone position step in one sample that is a GPS reset
SPIKE_DEG = 5      # swing step in one sample that is a false detection
SPIKE_S = 1.5      # how long a false ring lasts before the track comes back
WIN_S = 15.0       # a corner hold, from the moment the reference stops
SETTLE_M = 0.3     # the band the payload has to stay inside to count as settled
LEG_S = 27.5       # a whole leg, transit and hold, from the moment the reference moves


def _gapped(df, p):
    """
    The track with a gap at every logged glitch
    """
    d = df[["drone_px_meas", "drone_py_meas"]].to_numpy(float)
    a = np.degrees(df[["payload_alpha_x", "payload_alpha_y"]].to_numpy(float))
    t = df.cur_time.to_numpy(float)
    bad = np.zeros(len(df), bool)
    bad[1:] |= np.hypot(*np.diff(d, axis=0).T) > RESET_M
    spike = np.zeros(len(df), bool)
    spike[1:] = np.hypot(*np.diff(a, axis=0).T) > SPIKE_DEG
    edges = np.flatnonzero(spike)
    for i, j in zip(edges[:-1], edges[1:]):
        if t[j] - t[i] < SPIKE_S:
            spike[i:j + 1] = True
    bad |= spike
    bad[:-1] |= bad[1:]
    p = p.copy()
    p[bad] = np.nan

    return p


def track(sel):
    """
    One run's payload track and reference, both put on the reference's first
    sample, the northeast corner
    """
    s = catalog.resolve(sel)
    df = s.fl
    ref = df[["payload_px_ref", "payload_py_ref"]].to_numpy(float)
    origin = ref[0]
    p, _ = payload_enu(df, L=s.config.get("TETHER_LEN"))
    p = _gapped(df, p)

    return p[:, 0:2] - origin, ref - origin


def arrivals(sel):
    """
    Swing and payload error over every corner hold, on a clock that starts
    when the reference stops moving
    """
    s = catalog.resolve(sel)
    df = s.fl
    t = df.cur_time.to_numpy()
    fs = 1/np.median(np.diff(t))
    n = int(WIN_S*fs)
    p, ref = track(sel)
    err = np.hypot(*(p - ref).T)
    swing = np.degrees(np.hypot(df.payload_alpha_x,
                       df.payload_alpha_y).to_numpy())
    swing[np.isnan(p[:, 0])] = np.nan

    rv = np.hypot(np.gradient(ref[:, 0], t), np.gradient(ref[:, 1], t))
    idx = np.flatnonzero((rv > 0.05) & (t > 12))
    out = []
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        e = seg[-1]
        if len(seg) < 5*fs:
            continue
        # the last hold can run off the end of the log: keep what there is
        win = np.full((2, n), np.nan)
        k = min(n, len(t) - e)
        win[0, :k] = swing[e:e + k]
        win[1, :k] = err[e:e + k]
        out.append(win)

    return np.arange(n)/fs, out


def _legend(ax, size=LEGEND_SIZE, loc="best"):
    ax.legend(loc=loc, framealpha=1.0, edgecolor=C_INK,
              facecolor="white", fancybox=False, fontsize=size,
              labelcolor=C_INK)


def leg_errors(sel):
    """
    Track error RMS for each leg of one run, and over the whole square. A leg
    runs from the moment its reference starts moving to the moment the next
    one does, so it carries the transit and the hold at its corner.
    """
    s = catalog.resolve(sel)
    t = s.fl.cur_time.to_numpy()
    p, ref = track(sel)
    err = np.hypot(*(p - ref).T)

    rv = np.hypot(np.gradient(ref[:, 0], t), np.gradient(ref[:, 1], t))
    idx = np.flatnonzero((rv > 0.05) & (t > 12))
    starts = [seg[0] for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
              if t[seg[-1]] - t[seg[0]] > 5]
    out = []
    for b, e in zip(starts, starts[1:] + [len(t)]):
        d = ref[min(b + 50, len(t) - 1)] - ref[b]
        heading = "ENWS"[int(np.round(np.arctan2(d[1], d[0])/(np.pi/2))) % 4]
        out.append((heading, np.sqrt(np.nanmean(err[b:e]**2))))
    square = np.sqrt(np.nanmean(err[starts[0]:]**2))

    return out, square


def print_rms():
    legs = {}
    print(f"{'Run':<18} {'':<10} {'Leg':>4} {'':>2} {'Track Error RMS':>16}")
    for sel, kind in RUNS:
        out, square = leg_errors(sel)
        for k, (heading, e) in enumerate(out, 1):
            legs.setdefault(kind, []).append(e)
            print(f"{sel:<18} {kind:<10} {k:>4} {heading:>2} {e:>14.2f} m")
        print(f"{sel:<18} {kind:<10} {'Square':>7} {square:>14.2f} m")
        print()
    for kind, v in legs.items():
        print(f"{'Mean Of ' + str(len(v)) +
              ' Legs':<18} {kind:<10} {'':>7} {np.mean(v):>14.2f} m")
    print()
    print(f"{'Per-Leg Spread':<18} {'':<10} {'Mean':>7} {'Std':>7} {'Median':>7} "
          f"{'Min':>7} {'Max':>7}")
    for kind, v in legs.items():
        v = np.array(v)
        print(f"{'':<18} {kind:<10} {v.mean():>7.2f} {v.std(ddof=1):>7.2f} "
              f"{np.median(v):>7.2f} {v.min():>7.2f} {v.max():>7.2f}   m")


def phase_stats(sel):
    """
    Each leg split at the moment its reference reaches the corner: the transit
    while the reference moves, and the dwell from then until the next leg's
    reference starts, or the log ends. Horizontal payload position RMSE and
    swing angle, peak and RMS, for both.
    """
    s = catalog.resolve(sel)
    df = s.fl
    t = df.cur_time.to_numpy()
    p, ref = track(sel)
    err = np.hypot(*(p - ref).T)
    swing_xy = np.degrees(
        df[["payload_alpha_x", "payload_alpha_y"]].to_numpy(float))
    swing_xy[np.isnan(p[:, 0])] = np.nan
    swing = np.hypot(swing_xy[:, 0], swing_xy[:, 1])
    fs = 1/np.median(np.diff(t))
    T_n = 2*np.pi*np.sqrt(s.config["TETHER_LEN"]/9.81)

    rv = np.hypot(np.gradient(ref[:, 0], t), np.gradient(ref[:, 1], t))
    idx = np.flatnonzero((rv > 0.05) & (t > 12))
    segs = [seg for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            if t[seg[-1]] - t[seg[0]] > 5]

    def rms(a):
        return np.sqrt(np.nanmean(a**2))

    out = []
    for k, seg in enumerate(segs):
        b, e = seg[0], seg[-1] + 1
        end = segs[k + 1][0] if k + 1 < len(segs) else len(t)
        # settled once the error leaves the band for the last time, provided it
        # does so before the dwell runs out
        td = t[e:end] - t[e]
        above = np.flatnonzero(err[e:end] > SETTLE_M)
        settle = (0.0 if not len(above) else
                  td[above[-1] + 1] if above[-1] + 1 < len(td) else np.nan)

        # what is still swinging over the last pendulum period of the dwell:
        # half the peak to peak about the mean, along the direction it swings
        tail = slice(end - int(T_n*fs), end)
        err_tail = err[tail]
        err_tail = err_tail[np.isfinite(err_tail)]
        amp = []
        for a in (p[tail], np.radians(swing_xy[tail])):
            a = a[np.isfinite(a).all(1)]
            c = a - a.mean(0)
            along = c @ np.linalg.svd(c, full_matrices=False)[2][0]
            amp.append((along.max() - along.min())/2)

        out.append(dict(transit_rmse=rms(err[b:e]), dwell_rmse=rms(err[e:end]),
                        transit_peak=np.nanmax(swing[b:e]), transit_rms=rms(swing[b:e]),
                        dwell_peak=np.nanmax(swing[e:end]), dwell_rms=rms(swing[e:end]),
                        transit_s=t[e - 1] - t[b], dwell_s=t[end - 1] - t[e],
                        resid_m=amp[0], resid_deg=np.degrees(amp[1]), settle_s=settle,
                        p2p_m=2*amp[0], err_max=err_tail.max()))

    return out


def print_phases():
    cols = ("transit_rmse", "dwell_rmse", "transit_peak", "transit_rms",
            "dwell_peak", "dwell_rms")
    heads = ("Transit RMSE", "Dwell RMSE", "Transit Peak", "Transit RMS",
             "Dwell Peak", "Dwell RMS")
    units = ("m", "m", "deg", "deg", "deg", "deg")
    print(f"{'Run':<18} {'':<10} {'Leg':>4} " + " ".join(f"{h:>13}" for h in heads)
          + f" {'Dwell':>7}")
    legs = {}
    for sel, kind in RUNS:
        for k, d in enumerate(phase_stats(sel), 1):
            legs.setdefault(kind, []).append(d)
            print(f"{sel:<18} {kind:<10} {k:>4} "
                  + " ".join(f"{d[c]:>9.2f} {u_:<3}" for c,
                             u_ in zip(cols, units))
                  + f" {d['dwell_s']:>5.1f} s")
        print()
    print(f"{'Mean ± Std':<18} {'':<10} {'':>4} " +
          " ".join(f"{h:>13}" for h in heads))
    for kind, v in legs.items():
        print(f"{'Of ' + str(len(v)) + ' Legs':<18} {kind:<10} {'':>4} "
              + " ".join(f"{np.mean([d[c] for d in v]):>6.2f}±{np.std([d[c] for d in v], ddof=1):<4.2f} {u_:<1}"
                         for c, u_ in zip(cols, ("m", "m", "°", "°", "°", "°"))))
    for kind, v in legs.items():
        print(f"{'Largest Peak':<18} {kind:<10} transit {max(d['transit_peak'] for d in v):.2f} deg, "
              f"dwell {max(d['dwell_peak'] for d in v):.2f} deg")

    print()
    print(f"End of dwell, over the last pendulum period. Settled = inside "
          f"{SETTLE_M:.2f} m for the rest of the dwell")
    print(f"{'Run':<18} {'':<10} {'Leg':>4} {'Residual Amp':>13} {'':>9} {
          'Peak To Peak':>13} {'Max Error':>10} {'Settle Time':>12}")
    for sel, kind in RUNS:
        for k, d in enumerate(phase_stats(sel), 1):
            settle = (f"{d['settle_s']:>10.1f} s" if np.isfinite(d["settle_s"])
                      else f"{'not settled':>12}")
            print(f"{sel:<18} {kind:<10} {k:>4} {d['resid_m']:>11.2f} m "
                  f"{d['resid_deg']:>6.2f} deg {d['p2p_m']:>11.2f} m {d['err_max']:>8.2f} m {settle}")
        print()
    for kind, v in legs.items():
        amp_m = np.array([d["resid_m"] for d in v])
        amp_deg = np.array([d["resid_deg"] for d in v])
        p2p = np.array([d["p2p_m"] for d in v])
        err_max = np.array([d["err_max"] for d in v])
        st = np.array([d["settle_s"] for d in v])
        ok = np.isfinite(st)
        print(f"{'Of ' + str(len(v)) + ' Legs':<18} {kind:<10} residual "
              f"{amp_m.mean():.2f}±{amp_m.std(ddof=1):.2f} m, "
              f"{amp_deg.mean():.2f}±{amp_deg.std(
                  ddof=1):.2f} deg; peak to peak "
              f"{p2p.mean():.2f}±{p2p.std(ddof=1):.2f} m; max error "
              f"{err_max.mean():.2f}±{err_max.std(ddof=1):.2f} m; settled "
              f"{ok.sum()} of {len(st)} legs"
              + (f", {np.mean(st[ok]):.1f}±{np.std(st[ok], ddof=1):.1f} s" if ok.sum() > 1
                 else (f", {st[ok][0]:.1f} s" if ok.sum() == 1 else "")))


def detection(sel):
    """
    How often the tracker saw the payload while the square was flown.

    A frame counts as detected when the tracker reported a ring at all, and
    as a valid detection when that ring also sat within the range gate of the
    tether length. The longest dropout is the longest run of frames without a
    valid detection.
    """
    s = catalog.resolve(sel)
    fl = s.fl
    t = fl.cur_time.to_numpy()
    L = s.config["TETHER_LEN"]
    ref = fl[["payload_px_ref", "payload_py_ref"]].to_numpy(float)
    rv = np.hypot(np.gradient(ref[:, 0], t), np.gradient(ref[:, 1], t))
    t0 = t[np.flatnonzero((rv > 0.05) & (t > 12))[0]]

    frames = s.poses.groupby("frame").first()
    t_cam = frames.time_s.to_numpy(float) - s.pose_offset
    k = (t_cam >= t0) & (t_cam <= t[-1])
    t_cam = t_cam[k]
    seen = np.isfinite(frames.u_px.to_numpy(float))[k]
    rng = frames.range_m.to_numpy(float)[k]
    valid = seen & (np.abs(rng - L) <= STICK_RANGE_TOL*L)

    longest, last = 0.0, t_cam[0]
    for tc, v in zip(t_cam, valid):
        if v:
            last = tc
        longest = max(longest, tc - last)

    return dict(frames=len(t_cam), fps=(len(t_cam) - 1)/(t_cam[-1] - t_cam[0]),
                seen=seen.mean(), valid=valid.mean(), longest=longest)


def print_detection():
    print(f"{'Run':<18} {'':<10} {'Frames':>7} {'Rate':>7} {'Detected':>9} "
          f"{'Valid':>7} {'Longest Dropout':>16}")
    rows = []
    for sel, kind in RUNS:
        d = detection(sel)
        rows.append(d)
        print(f"{sel:<18} {kind:<10} {d['frames']:>7d} {d['fps']:>5.1f}Hz "
              f"{100*d['seen']:>8.2f}% {100*d['valid']:>6.2f}% {d['longest']:>14.2f} s")
    n = sum(d["frames"] for d in rows)
    seen = sum(d["seen"]*d["frames"] for d in rows)/n
    valid = sum(d["valid"]*d["frames"] for d in rows)/n
    print(f"{'All 4 Runs':<18} {'':<10} {n:>7d} {'':>7} {100*seen:>8.2f}% "
          f"{100*valid:>6.2f}% {max(d['longest'] for d in rows):>14.2f} s")


def tracks_figure(save=None):
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    seen = set()
    ref_drawn = None
    for sel, kind in RUNS:
        p, ref = track(sel)
        ax.plot(p[:, 0], p[:, 1], color=COLORS[kind], lw=1.2, alpha=0.85,
                label=None if kind in seen else NAMES[kind])
        seen.add(kind)
        ref_drawn = ref if ref_drawn is None else ref_drawn
    ax.plot(ref_drawn[:, 0], ref_drawn[:, 1], ls=(0, (6, 4)), lw=1.5,
            color="#8A8F94", zorder=0, label="Reference")
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.06)
    _frame(ax)
    _legend(ax)
    fig.tight_layout()
    if save:
        fig.savefig(save/"icra_square_tracks.pdf", dpi=300, facecolor="white")
        fig.savefig(save/"icra_square_tracks.png", dpi=200, facecolor="white")

    return fig


def _settling(save, k, ylabel, ylim, stem):
    """
    One arrival panel: column k of every hold, faint, with the medians bold
    """
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    legs = {"payload": [], "ardupilot": []}
    for sel, kind in RUNS:
        tt, out = arrivals(sel)
        legs[kind] += out
    # ardupilot first so the payload median draws over it, then the legend is
    # put back in the order every other figure uses
    handles = {}
    for kind in ("ardupilot", "payload"):
        arr = np.array(legs[kind])[:, k]
        for row in arr:
            ax.plot(tt, row, color=COLORS[kind], lw=0.6, alpha=0.2)
        handles[kind], = ax.plot(tt, np.nanmedian(arr, axis=0),
                                 color=COLORS[kind], lw=2.2,
                                 label=f"{NAMES[kind]} ({len(arr)} Legs)")
    ax.set_xlabel("Time After Leg End [s]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, WIN_S)
    ax.set_ylim(0, ylim)
    _frame(ax)
    ordered = [handles["payload"], handles["ardupilot"]]
    ax.legend(ordered, [h.get_label() for h in ordered], loc="upper right",
              framealpha=1.0, edgecolor=C_INK, facecolor="white",
              fancybox=False, fontsize=LEGEND_SIZE - 2, labelcolor=C_INK)
    fig.tight_layout()
    if save:
        fig.savefig(save/f"{stem}.pdf", dpi=300, facecolor="white")
        fig.savefig(save/f"{stem}.png", dpi=200, facecolor="white")

    return fig


def leg_windows(sel):
    """
    Payload position error, east north up, over every leg, on a clock that
    starts when the reference begins to move, and when the reference reached
    its corner
    """
    s = catalog.resolve(sel)
    df = s.fl
    t = df.cur_time.to_numpy()
    fs = 1/np.median(np.diff(t))
    n = int(LEG_S*fs)
    p, _ = payload_enu(df, L=s.config.get("TETHER_LEN"))
    p = _gapped(df, p)
    ref3 = df[["payload_px_ref", "payload_py_ref",
               "payload_pz_ref"]].to_numpy(float)
    err = p[:, 0:3] - ref3
    ref = ref3[:, 0:2]

    rv = np.hypot(np.gradient(ref[:, 0], t), np.gradient(ref[:, 1], t))
    idx = np.flatnonzero((rv > 0.05) & (t > 12))
    out, arrive = [], []
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        b, e = seg[0], seg[-1]
        if len(seg) < 5*fs:
            continue
        win = np.full((n, 3), np.nan)
        k = min(n, len(t) - b)
        win[:k] = err[b:b + k]
        out.append(win)
        arrive.append(t[e] - t[b])

    return np.arange(n)/fs, out, arrive


def leg_error_figure(save=None):
    """
    Payload position error through a leg and its hold: every leg faint, the
    mean bold
    """
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    legs = {"payload": [], "ardupilot": []}
    arrive = []
    for sel, kind in RUNS:
        tt, out, ta = leg_windows(sel)
        legs[kind] += out
        arrive += ta

    handles = {}
    for kind in ("ardupilot", "payload"):
        arr = np.linalg.norm(np.array(legs[kind]), axis=2)
        for row in arr:
            ax.plot(tt, row, color=COLORS[kind], lw=0.6, alpha=0.2)
        handles[kind], = ax.plot(tt, np.nanmean(arr, axis=0), color=COLORS[kind],
                                 lw=2.2, label=f"{NAMES[kind]} ({len(arr)} Legs)")
    handles["arrive"] = ax.axvline(np.median(arrive), color="#8A8F94",
                                   ls=(0, (6, 4)), lw=1.2, zorder=0,
                                   label="Reference Reaches Corner")

    ax.set_xlabel("Time Since Leg Start [s]")
    ax.set_ylabel("Total Payload Position Error [m]")
    ax.set_xlim(0, LEG_S)
    ax.set_ylim(0, 2.6)
    _frame(ax)
    ordered = [handles["payload"], handles["ardupilot"], handles["arrive"]]
    ax.legend(ordered, [h.get_label() for h in ordered], loc="upper right",
              framealpha=1.0, edgecolor=C_INK, facecolor="white",
              fancybox=False, fontsize=LEGEND_SIZE - 2, labelcolor=C_INK)

    fig.tight_layout()
    if save:
        fig.savefig(save/"icra_leg_error.pdf", dpi=300, facecolor="white")
        fig.savefig(save/"icra_leg_error.png", dpi=200, facecolor="white")

    return fig


def square_windows(sel):
    """
    Payload position error, east north up, over the whole square, on a clock
    that starts when the first leg's reference begins to move, and when each
    leg started on that clock
    """
    s = catalog.resolve(sel)
    df = s.fl
    t = df.cur_time.to_numpy()
    p, _ = payload_enu(df, L=s.config.get("TETHER_LEN"))
    p = _gapped(df, p)
    ref3 = df[["payload_px_ref", "payload_py_ref",
               "payload_pz_ref"]].to_numpy(float)
    err = p[:, 0:3] - ref3

    rv = np.hypot(np.gradient(ref3[:, 0], t), np.gradient(ref3[:, 1], t))
    idx = np.flatnonzero((rv > 0.05) & (t > 12))
    starts = [seg[0] for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
              if t[seg[-1]] - t[seg[0]] > 5]
    b = starts[0]

    return t[b:] - t[b], err[b:], [t[i] - t[b] for i in starts]


def leg_axis_error_figure(save=None):
    """
    Payload position error east and north through each leg of the square, one
    panel per leg, averaged over the runs of each controller.

    Every run flies the legs in the same order, so leg k of one run lines up
    with leg k of the next; the size of each component is averaged. The panels
    share both axes so they read against each other.
    """
    legs = {"payload": [], "ardupilot": []}
    arrive = []
    for sel, kind in RUNS:
        tt, out, ta = leg_windows(sel)
        legs[kind].append(out)
        arrive += ta

    headings = ("South", "West", "North", "East")
    means = {kind: np.nanmean(np.abs(np.array(v)), axis=0)
             for kind, v in legs.items()}                  # (legs, samples, 3)
    top = 1.12*max(np.nanmax(m[:, :, 0:2]) for m in means.values())

    fig, axes = plt.subplots(4, 1, figsize=(
        6.4, 9.0), sharex=True, sharey=True)
    handles = {}
    for leg, (ax, heading) in enumerate(zip(axes, headings)):
        for kind in ("ardupilot", "payload"):
            for k, axis, ls in ((0, "East", "-"), (1, "North", (0, (4, 3)))):
                handles[kind, axis], = ax.plot(
                    tt, means[kind][leg, :, k], color=COLORS[kind], lw=1.4,
                    ls=ls, label=f"{NAMES[kind]}, {axis}")
        handles["arrive"] = ax.axvline(np.median(arrive), color="#8A8F94",
                                       ls=(0, (6, 4)), lw=1.0, zorder=0,
                                       label="Reference Reaches Corner")
        ax.text(0.995, 0.95, f"{heading} Leg", transform=ax.transAxes,
                ha="right", va="top", fontsize=LEGEND_SIZE - 3, color=C_INK,
                bbox=dict(facecolor="white", edgecolor=C_INK, lw=0.8, pad=3))
        ax.set_xlim(0, LEG_S)
        ax.set_ylim(0, top)
        _frame(ax)

    axes[-1].set_xlabel("Time Since Leg Start [s]")
    fig.supylabel("Payload Position Error [m]")
    # two columns fill top to bottom: payload down the left, baseline down the right
    ordered = [handles["payload", "East"], handles["payload", "North"],
               handles["arrive"],
               handles["ardupilot", "East"], handles["ardupilot", "North"]]
    axes[0].legend(ordered, [h.get_label() for h in ordered], loc="lower center",
                   bbox_to_anchor=(0.5, 1.03), ncol=2, framealpha=1.0,
                   edgecolor=C_INK, facecolor="white", fancybox=False,
                   fontsize=LEGEND_SIZE - 3, labelcolor=C_INK)

    fig.tight_layout()
    if save:
        fig.savefig(save/"icra_leg_error_en.pdf", dpi=300, facecolor="white")
        fig.savefig(save/"icra_leg_error_en.png", dpi=200, facecolor="white")

    return fig


def dwell_errors(sel):
    """
    Horizontal payload position error through each dwell, on a clock that
    starts when the leg's reference reaches its corner
    """
    s = catalog.resolve(sel)
    t = s.fl.cur_time.to_numpy()
    p, ref = track(sel)
    err = np.hypot(*(p - ref).T)

    rv = np.hypot(np.gradient(ref[:, 0], t), np.gradient(ref[:, 1], t))
    idx = np.flatnonzero((rv > 0.05) & (t > 12))
    segs = [seg for seg in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)
            if t[seg[-1]] - t[seg[0]] > 5]
    out = []
    for k, seg in enumerate(segs):
        e = seg[-1] + 1
        end = segs[k + 1][0] if k + 1 < len(segs) else len(t)
        out.append((t[e:end] - t[e], err[e:end]))

    return out


def settle_time(td, err, band):
    """
    When the error leaves the band for the last time, or NaN if it is still
    outside at the end of the dwell
    """
    above = np.flatnonzero(err > band)
    if not len(above):
        return 0.0
    if above[-1] + 1 >= len(td):
        return np.nan

    return td[above[-1] + 1]


def settle_figure(save=None):
    """
    Settling against the width of the band it is judged by: how many legs
    settle, and how long they take, with a leg that never settles counted as
    the whole dwell
    """
    bands = np.round(np.arange(0.10, 1.001, 0.01), 2)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    handles = {}
    for kind in ("ardupilot", "payload"):
        windows = [w for sel, k in RUNS if k ==
                   kind for w in dwell_errors(sel)]
        st = np.array([[settle_time(td, err, b) for b in bands]
                      for td, err in windows])
        dwell = np.array([td[-1] for td, _ in windows])
        censored = np.where(np.isnan(st), dwell[:, None], st)

        handles[kind], = axes[0].plot(bands, 100*np.isfinite(st).mean(axis=0),
                                      color=COLORS[kind], lw=2.0,
                                      label=f"{NAMES[kind]} ({len(windows)} Legs)")
        for row in censored:
            axes[1].plot(bands, row, color=COLORS[kind], lw=0.6, alpha=0.2)
        axes[1].plot(bands, censored.mean(axis=0), color=COLORS[kind], lw=2.0)

    for ax in axes:
        ax.axvline(SETTLE_M, color="#8A8F94",
                   ls=(0, (6, 4)), lw=1.0, zorder=0)
        _frame(ax)
    axes[0].set_ylabel("Legs Settled [%]")
    axes[0].set_ylim(0, 105)
    axes[1].set_ylabel("Settling Time [s]")
    axes[1].set_ylim(0, WIN_S + 0.5)
    axes[1].set_xlabel("Settling Band [m]")
    axes[1].set_xlim(bands[0], bands[-1])
    ordered = [handles["payload"], handles["ardupilot"]]
    axes[0].legend(ordered, [h.get_label() for h in ordered], loc="lower right",
                   framealpha=1.0, edgecolor=C_INK, facecolor="white",
                   fancybox=False, fontsize=LEGEND_SIZE - 2, labelcolor=C_INK)

    fig.tight_layout()
    if save:
        fig.savefig(save/"icra_settle_sweep.pdf", dpi=300, facecolor="white")
        fig.savefig(save/"icra_settle_sweep.png", dpi=200, facecolor="white")

    return fig


def position_figure(save=None):
    return _settling(save, 1, "Payload Position Error [m]", 2,
                     "icra_waypoint_position")


def swing_figure(save=None):
    return _settling(save, 0, "Payload Swing [deg]", 16,
                     "icra_waypoint_swing")


def stick_guided(sel=STICK_RUN):
    """
    The payload states and their reference while the aid was flying, with the
    swing re-estimated offline.

    The tracker ran a +-80% range gate, so a ring seen at 1.5 m on a 6.2 m
    tether reached the onboard filter and threw the estimate. Replaying the
    filter with those frames dropped takes the worst swing over this stretch
    from 32 deg to 11 deg. Everything is measured from the reference's first
    sample, so the pilot starts at the origin.
    """
    s = catalog.resolve(sel)
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
    t_est, p_est = ekf_plots.estimated_payload_enu(records, fl, L)

    # the payload rides the drone, leaning at the swing rate the filter saw
    t = fl.cur_time.to_numpy()
    v_drone = fl[["drone_vx_meas", "drone_vy_meas",
                  "drone_vz_meas"]].to_numpy(float)
    rate = np.array([[r["alpha_dot_x"], r["alpha_dot_y"], 0] for r in records])
    v_est = np.column_stack([np.interp(t_est, t, v_drone[:, k])
                             for k in range(3)]) + L*rate

    guided = (fl.echoed_mode.astype(str) == "GUIDED").to_numpy()
    lo, hi = t[guided].min(), t[guided].max()
    k = (t_est >= lo) & (t_est <= hi)
    p_ref = fl[["payload_px_ref", "payload_py_ref", "payload_pz_ref"]].to_numpy(float)[
        guided]
    v_ref = fl[["payload_vx_ref", "payload_vy_ref", "payload_vz_ref"]].to_numpy(float)[
        guided]
    origin = p_ref[0]

    return dict(t_est=t_est[k], p_est=p_est[k] - origin, v_est=v_est[k],
                t_ref=t[guided], p_ref=p_ref - origin, v_ref=v_ref)


def stick_figure(save=None):
    """
    The piloting aid: where the payload went against the pilot's reference.

    Drawn to scale and cropped to the flight, which ran about twice as far
    east as it did north, so the panel comes out long and shallow.
    """
    d = stick_guided()
    fig, ax = plt.subplots(figsize=(7.4, 3.4))

    ax.plot(d["p_ref"][:, 0], d["p_ref"][:, 1], ls=(0, (6, 4)), lw=1.5,
            color="#8A8F94", zorder=0, label="Pilot Reference")
    ax.plot(d["p_est"][:, 0], d["p_est"][:, 1], color=C_PAYLOAD_CTRL, lw=1.2,
            alpha=0.85, label="Payload")

    pad = 0.6
    x = np.concatenate([d["p_est"][:, 0], d["p_ref"][:, 0]])
    y = np.concatenate([d["p_est"][:, 1], d["p_ref"][:, 1]])
    ax.set_xlim(x.min() - pad, x.max() + pad)
    ax.set_ylim(y.min() - pad, y.max() + pad)
    ax.set_xlabel("East [m]")
    ax.set_ylabel("North [m]")
    ax.set_aspect("equal", adjustable="box")
    _frame(ax)
    _legend(ax, LEGEND_SIZE - 2, loc="lower left")

    fig.tight_layout()
    if save:
        fig.savefig(save/"icra_stick.pdf", dpi=300, facecolor="white")
        fig.savefig(save/"icra_stick.png", dpi=200, facecolor="white")

    return fig


def stick_states_figure(save=None):
    """
    Payload position and velocity against the reference, over the stretch the
    aid was flying.

    Time against metres has no aspect ratio to set, so the panels are simply
    made wider than they are tall and the pair squared off.
    """
    d = stick_guided()
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 6.2), sharex=True)
    comps = ((0, "East", C_EAST), (1, "North", C_NORTH), (2, "Up", C_UP))

    for k, name, color in comps:
        axes[0].plot(d["t_ref"], d["p_ref"][:, k], color=color, lw=1.0,
                     ls=(0, (4, 3)), alpha=0.8)
        axes[0].plot(d["t_est"], d["p_est"][:, k], color=color, lw=1.3,
                     label=name)
        axes[1].plot(d["t_ref"], d["v_ref"][:, k], color=color, lw=1.0,
                     ls=(0, (4, 3)), alpha=0.8)
        axes[1].plot(d["t_est"], d["v_est"][:, k], color=color, lw=1.3,
                     label=name)

    axes[0].plot([], [], color=C_INK, lw=1.0,
                 ls=(0, (4, 3)), label="Reference")
    axes[0].set_ylabel("Payload Position [m]")
    axes[1].set_ylabel("Payload Velocity [m/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_xlim(d["t_ref"].min(), d["t_ref"].max())
    for ax in axes:
        ax.axhline(0, color="#8A8F94", lw=0.8, zorder=0)
        _frame(ax)
    _legend(axes[0], LEGEND_SIZE - 2)

    fig.tight_layout()
    if save:
        fig.savefig(save/"icra_stick_states.pdf", dpi=300, facecolor="white")
        fig.savefig(save/"icra_stick_states.png", dpi=200, facecolor="white")

    return fig


if __name__ == '__main__':
    out = None
    if len(sys.argv) > 1:
        out = Path(sys.argv[1]).expanduser()
        out.mkdir(parents=True, exist_ok=True)
    print_rms()
    print()
    print_phases()
    print()
    print_detection()
    tracks_figure(out)
    position_figure(out)
    swing_figure(out)
    leg_error_figure(out)
    leg_axis_error_figure(out)
    settle_figure(out)
    stick_figure(out)
    stick_states_figure(out)
    plt.show()
