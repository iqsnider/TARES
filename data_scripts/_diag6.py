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
win = cv2.createHanningWindow((900, 900), cv2.CV_32F)
def patch(g):
    h, w = g.shape
    return g[h//2-450:h//2+450, w//2-450:w//2+450]

# legs: (name, first frame, last frame) taken from the reference ramps
legs = [("south", 820, 1400), ("west", 2180, 2760), ("north", 3450, 4000)]
step = 6
need = sorted({f for _, a, b in legs for f in range(a, b + 1, step)})
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

for name, a, b in legs:
    fs = [f for f in range(a, b + 1, step) if f in grab and f in by_frame]
    total = np.zeros(2)
    for f0, f1 in zip(fs, fs[1:]):
        sh, resp = cv2.phaseCorrelate(patch(grab[f0])*win, patch(grab[f1])*win)
        total += sh
    k0, k1 = by_frame[fs[0]], by_frame[fs[-1]]
    move = p_drone[k1] - p_drone[k0]
    # predicted: same ground point, attitude held at the leg's first frame
    T = recs[k0]["T_IB"]
    gnd = np.array([p_drone[k0][0], p_drone[k0][1], 0.0])
    u0 = E.project_enu(gnd, p_drone[k0], T, filt, K, D)[0]
    u1 = E.project_enu(gnd, p_drone[k1], T, filt, K, D)[0]
    print(f"{name:6s} drone moved ENU {np.round(move,2)}")
    print(f"       observed ground shift {np.round(total,1)} px")
    print(f"       project_enu predicts  {np.round(u1-u0,1)} px")
    ang = lambda v: np.degrees(np.arctan2(v[1], v[0]))
    print(f"       angle observed {ang(total):7.1f}   predicted {ang(u1-u0):7.1f}"
          f"   diff {((ang(u1-u0)-ang(total)+180) % 360) - 180:7.1f} deg")
