"""
Paired scoring of payload-controller squares against the ardupilot baseline.

    uv run python score_squares.py 09112026

Per run: swing decay rate at the corner holds (log decrement over the peaks
of the swing after each leg ends), rms swing over the last 8 s of each hold,
mean payload error against the commanded corner during those 8 s, and the
whole-flight rms swing for reference. Uses the swing the aircraft logged, so
the camera geometry must be right for the day.
"""
import sys
sys.path.insert(0, 'data_scripts')
import numpy as np, pandas as pd
from scipy.signal import find_peaks
import catalog

G = 9.81
TAIL = 8.0          # [s] scored at the end of each hold
MIN_HOLD = 10.0     # [s] shorter stills are not corner holds

def score(s):
    fl = s.fl
    L = s.config["TETHER_LEN"]
    t = fl.cur_time.to_numpy(); fs = 1/np.median(np.diff(t))
    a = fl[["payload_alpha_x", "payload_alpha_y"]].to_numpy(float)
    px = fl.drone_px_meas.to_numpy() + L*a[:, 0]
    py = fl.drone_py_meas.to_numpy() + L*a[:, 1]
    rx, ry = fl.payload_px_ref.to_numpy(float), fl.payload_py_ref.to_numpy(float)
    rv = np.hypot(np.gradient(rx, t), np.gradient(ry, t))
    hold = (rv < 0.05) & (t > 12)
    idx = np.flatnonzero(hold)
    T_n = 2*np.pi*np.sqrt(L/G)
    sig, tail_sw, tail_err, n_holds = [], [], [], 0
    for p in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        if len(p) < MIN_HOLD*fs:
            continue
        n_holds += 1
        seg = a[p] - np.nanmean(a[p], axis=0)
        if np.isfinite(seg).all():
            ax = np.linalg.svd(seg, full_matrices=False)[2][0]
            sgn = seg @ ax
            pk, _ = find_peaks(np.abs(sgn), distance=int(0.4*T_n*fs))
            if len(pk) >= 4:
                sig.append(-np.polyfit(t[p][pk], np.log(np.abs(sgn[pk])), 1)[0])
        tl = p[-int(TAIL*fs):]
        tail_sw.append(np.degrees(np.sqrt(np.nanmean(a[tl, 0]**2 + a[tl, 1]**2))))
        tail_err.append(np.nanmean(np.hypot(px[tl] - rx[tl], py[tl] - ry[tl])))
    m = t > 10
    whole = np.degrees(np.sqrt(np.nanmean(a[m, 0]**2 + a[m, 1]**2)))
    return dict(holds=n_holds, sigma=np.median(sig) if sig else np.nan,
                tail_sw=np.mean(tail_sw) if tail_sw else np.nan,
                tail_err=np.mean(tail_err) if tail_err else np.nan, whole=whole)

date = sys.argv[1]
pool = [s for s in catalog.select(date) if len(s.fl) > 3000 and "icra_test" in s.label]
groups = {"ardupilot": [], "payload": []}
print("%-40s %5s %10s %11s %11s %10s" % ("session", "holds", "decay 1/s", "tail swing", "tail err", "whole rms"))
for s in sorted(pool, key=lambda x: x.id):
    kind = "ardupilot" if "ardupilot" in s.label else "payload"
    r = score(s)
    groups[kind].append(r)
    ctrl = s.config.get("CONTROLLER") or "ardupilot"
    tc = s.config.get("LQI_PAYLOAD_TUNING_CONST", float("nan")) if kind == "payload" else float("nan")
    print("%-40s %5d %10.3f %9.2f deg %9.2f m %8.2f deg   %s%s" % (
        s.label, r["holds"], r["sigma"], r["tail_sw"], r["tail_err"], r["whole"], ctrl[-3:] if kind == "payload" else "",
        "  tc %.1f" % tc if kind == "payload" else ""))
print()
for kind, rs in groups.items():
    if not rs:
        continue
    print("%-10s n=%d   decay %.3f 1/s   tail swing %.2f deg   tail err %.2f m   whole rms %.2f deg" % (
        kind, len(rs), np.nanmedian([r["sigma"] for r in rs]), np.nanmean([r["tail_sw"] for r in rs]),
        np.nanmean([r["tail_err"] for r in rs]), np.nanmean([r["whole"] for r in rs])))
print()
print("decay > 0 is damping. 0909 baseline holds: 0.04 1/s median (payload drag only). Corrected LQI predicted 0.2 to 0.4 1/s.")
