import cv2, numpy as np
import catalog, ekf_plots as E
import sim.estimation.ekf as ekfm
import sim.estimation.pre_process as pp

s = catalog.resolve("09092026.132352")
recs = E.logged_records(s)
K, D = E._intrinsics()
geom = pp.Geometry.from_snapshot(s.config)
filt = E.make_ekf(0, 0, 0, 0, 0, 0, geom=geom, source=ekfm.SOURCE_ARUCO,
                  L=s.config.get("TETHER_LEN"))
p_drone, _ = E.drone_states(s.fl, recs)
by_frame = {r["frame"]: k for k, r in enumerate(recs)}

VID = "../data/test_09092026/ardupilot_icra_test_20260909_132351/recording.avi"
gap = 24                       # half a second
spans = [(800, 1450), (2150, 2800), (3400, 4050), (4700, 5350)]
pairs = [(f, f + gap) for a, b in spans for f in range(a, b, gap)]
need = sorted({f for p in pairs for f in p})

grab = {}
cap = cv2.VideoCapture(VID)
i = 0
while True:
    ok, fr = cap.read()
    if not ok or i > need[-1]:
        break
    if i in need:
        grab[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    i += 1
cap.release()

win = cv2.createHanningWindow((1000, 1000), cv2.CV_32F)
def patch(g):
    h, w = g.shape
    return g[h//2-500:h//2+500, w//2-500:w//2+500]

ang = lambda v: np.degrees(np.arctan2(v[1], v[0]))
diffs = []
for a, b in pairs:
    if a not in grab or b not in grab or a not in by_frame or b not in by_frame:
        continue
    sh, resp = cv2.phaseCorrelate(patch(grab[a])*win, patch(grab[b])*win)
    ka, kb = by_frame[a], by_frame[b]
    T = recs[ka]["T_IB"]
    gnd = np.array([p_drone[ka][0], p_drone[ka][1], 0.0])
    u0 = E.project_enu(gnd, p_drone[ka], T, filt, K, D)[0]
    u1 = E.project_enu(gnd, p_drone[kb], T, filt, K, D)[0]
    pr = u1 - u0
    if resp < 0.35 or np.hypot(*sh) < 20 or np.hypot(*pr) < 20:
        continue
    d = ((ang(pr) - ang(sh) + 180) % 360) - 180
    diffs.append(d)
    print(f"{a:5d} obs {np.round(sh,0)} |{np.hypot(*sh):6.1f}|   "
          f"pred {np.round(pr,0)} |{np.hypot(*pr):6.1f}|   diff {d:7.1f} deg")

diffs = np.array(diffs)
print(f"\n{len(diffs)} samples, median angle error "
      f"{np.median(np.abs(diffs)):.1f} deg")
