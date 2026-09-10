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
p_drone, v_drone = E.drone_states(s.fl, recs)
by_frame = {r["frame"]: k for k, r in enumerate(recs)}

pairs = [(f, f + 12) for f in range(600, 5800, 200)]
need = sorted({f for p in pairs for f in p})
cap = cv2.VideoCapture("../data/test_09092026/ardupilot_icra_test_20260909_132351/recording.avi")
grabbed = {}
i = 0
while True:
    ok, fr = cap.read()
    if not ok or i > need[-1]:
        break
    if i in grabbed or i in need:
        grabbed[i] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    i += 1
cap.release()

win = cv2.createHanningWindow((800, 800), cv2.CV_32F)
def patch(g):
    h, w = g.shape
    return g[h//2-400:h//2+400, w//2-400:w//2+400]

obs, pred = [], []
for a, b in pairs:
    if a not in grabbed or b not in grabbed:
        continue
    if a not in by_frame or b not in by_frame:
        continue
    sh, resp = cv2.phaseCorrelate(patch(grabbed[a])*win, patch(grabbed[b])*win)
    if resp < 0.3 or np.hypot(*sh) < 5:
        continue
    ka, kb = by_frame[a], by_frame[b]
    ra, rb = recs[ka], recs[kb]
    # a world point on the ground straight under the drone at frame a
    gnd = np.array([p_drone[ka][0], p_drone[ka][1], 0.0])
    ua = E.project_enu(gnd, p_drone[ka], ra["T_IB"], filt, K, D)[0]
    ub = E.project_enu(gnd, p_drone[kb], rb["T_IB"], filt, K, D)[0]
    if not (np.isfinite(ua).all() and np.isfinite(ub).all()):
        continue
    obs.append(sh)
    pred.append(ub - ua)

obs = np.array(obs); pred = np.array(pred)
print(f"{len(obs)} usable pairs")
# least squares 2x2 mapping pred = M @ obs
M, *_ = np.linalg.lstsq(obs, pred, rcond=None)
M = M.T
print("M (pred = M @ obs):\n", np.round(M, 3))
u, sv, vt = np.linalg.svd(M)
R = u @ vt
print("scale:", np.round(sv, 3), " rotation deg:",
      round(np.degrees(np.arctan2(R[1, 0], R[0, 0])), 1))
for o, p in zip(obs[:12], pred[:12]):
    print("obs", np.round(o, 1), " pred", np.round(p, 1))
