import numpy as np
import catalog, ekf_plots as E
import sim.estimation.ekf as ekfm

s = catalog.resolve("09092026.132352")
recs = E.logged_records(s)
K, D = E._intrinsics()
import sim.estimation.pre_process as pp
geom = pp.Geometry.from_snapshot(s.config)
filt = E.make_ekf(0,0,0,0,0,0, geom=geom, source=ekfm.SOURCE_ARUCO,
                  L=s.config.get("TETHER_LEN"))
p_drone, v_drone = E.drone_states(s.fl, recs)
by_frame = {r["frame"]: k for k, r in enumerate(recs)}

fA, fB = 2400, 2496
kA, kB = by_frame[fA], by_frame[fB]
rA, rB = recs[kA], recs[kB]
print("t_flight A,B:", rA["t_flight"], rB["t_flight"])
print("p_drone A:", p_drone[kA], " B:", p_drone[kB])
print("v_drone A:", v_drone[kA])

# a fixed world point: directly under the drone at frame A, at reference height
gnd = np.array([p_drone[kA][0], p_drone[kA][1], 6.883192])
uvA = E.project_enu(gnd, p_drone[kA], rA["T_IB"], filt, K, D)[0]
uvB = E.project_enu(gnd, p_drone[kB], rB["T_IB"], filt, K, D)[0]
print("fixed ground point px A:", uvA, " B:", uvB, " delta:", uvB-uvA)
print("  (in 960-wide view, delta =", (uvB-uvA)*960/2304, ")")

# world unit vectors at reference depth
for name, d in [("+E", [1,0,0]), ("+N", [0,1,0])]:
    p = gnd + np.array(d, float)
    uv = E.project_enu(p, p_drone[kA], rA["T_IB"], filt, K, D)[0]
    print(f"  {name} 1 m ->", (uv-uvA), "px")

refpx = E.reference_pixels(s, recs, filt, K, D)
for f in (300, 1200, 2400, 2448, 2496, 3600, 4800):
    if f in refpx:
        runs, here, gap = refpx[f]
        print(f, "here_px", np.round(here,1), "gap_m", round(gap,3),
              "runs", [len(r) for r in runs])
