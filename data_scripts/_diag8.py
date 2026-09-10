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
legs = {"south": (900, 1300), "west": (2250, 2650), "north": (3550, 3950)}
lo, hi = 900, 3950

cap = cv2.VideoCapture(VID)
prev = None
flow = {}                              # frame -> chained displacement
pts = None
acc = np.zeros(2)
i = 0
mask = None
while True:
    ok, fr = cap.read()
    if not ok or i > hi:
        break
    if i >= lo:
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if mask is None:
            h, w = g.shape
            mask = np.zeros((h, w), np.uint8)
            mask[h//2-560:h//2+560, w//2-560:w//2+560] = 255
        if prev is not None:
            nxt, st, err = cv2.calcOpticalFlowPyrLK(prev, g, pts, None,
                                                    winSize=(31, 31), maxLevel=4)
            good = (st.ravel() == 1)
            if good.sum() >= 8:
                d = (nxt[good] - pts[good]).reshape(-1, 2)
                acc = acc + np.median(d, axis=0)
            pts = None
        if pts is None:
            pts = cv2.goodFeaturesToTrack(g, 300, 0.01, 20, mask=mask)
        flow[i] = acc.copy()
        prev = g
    i += 1
cap.release()

ang = lambda v: np.degrees(np.arctan2(v[1], v[0]))
for name, (a, b) in legs.items():
    obs = flow[b] - flow[a]
    ka, kb = by_frame[a], by_frame[b]
    T = recs[ka]["T_IB"]
    gnd = np.array([p_drone[ka][0], p_drone[ka][1], 0.0])
    u0 = E.project_enu(gnd, p_drone[ka], T, filt, K, D)[0]
    u1 = E.project_enu(gnd, p_drone[kb], T, filt, K, D)[0]
    pr = u1 - u0
    move = p_drone[kb] - p_drone[ka]
    d = ((ang(pr) - ang(obs) + 180) % 360) - 180
    print(f"{name:6s} drone ENU move {np.round(move,2)}")
    print(f"       observed ground {np.round(obs,0)} |{np.hypot(*obs):7.1f}| "
          f"ang {ang(obs):7.1f}")
    print(f"       predicted       {np.round(pr,0)} |{np.hypot(*pr):7.1f}| "
          f"ang {ang(pr):7.1f}   -> off by {d:7.1f} deg\n")
