"""
Airframe response to the commanded acceleration at the pendulum frequency,
read off a flight log. Run after the first payload flight tomorrow:

    uv run python airframe_check.py 09112026.HHMMSS

Gain near 1 and phase near -25 deg is what 0909 measured: fly tc 2.0.
Phase past -35 deg: stay at tc 4.0.
"""
import sys
sys.path.insert(0, 'data_scripts'); sys.path.insert(0, 'src')
import numpy as np, pandas as pd
from scipy.signal import welch, csd
import catalog
import sim.transformations as tf

G = 9.81
S = tf.T_ENU_from_NED()

s = catalog.resolve(sys.argv[1])
fl = s.fl
L = s.config["TETHER_LEN"]
wn = np.sqrt(G/L)
t = fl.cur_time.to_numpy()
fs = 1/np.median(np.diff(t))
u = np.column_stack([pd.to_numeric(fl.ux, errors="coerce"), pd.to_numeric(fl.uy, errors="coerce")]).astype(float)
live = (t > 10) & np.isfinite(u).all(1) & (fl.echoed_mode.astype(str) == "GUIDED").to_numpy()
u = u[live]
lean = np.array([G*(S @ tf.T_IB(r, p, y) @ S @ np.array([0, 0, 1.0]))[:2]
                 for r, p, y in zip(fl.drone_roll[live], fl.drone_pitch[live], fl.drone_yaw[live])])

Pxx = Pxy = Pyy = 0
for i in range(2):
    f, pxx = welch(u[:, i], fs, nperseg=1024)
    _, pxy = csd(u[:, i], lean[:, i], fs, nperseg=1024)
    _, pyy = welch(lean[:, i], fs, nperseg=1024)
    Pxx = Pxx + pxx; Pxy = Pxy + pxy; Pyy = Pyy + pyy
w = 2*np.pi*f
print(f"{s.id}  {s.label}   L {L:.1f} m, pendulum {wn:.2f} rad/s, {live.sum()/fs:.0f} s of closed-loop data")
print("%8s %8s %8s %6s" % ("rad/s", "gain", "phase", "coh"))
for we in (0.6, 0.8, 1.0, wn, 1.6, 2.0):
    k = np.argmin(abs(w - we))
    H = Pxy[k]/Pxx[k]
    coh = abs(Pxy[k])**2/(Pxx[k]*Pyy[k])
    mark = "  <- pendulum" if we == wn else ""
    print("%8.2f %8.2f %7.0f deg %5.2f%s" % (w[k], abs(H), np.degrees(np.angle(H)), coh, mark))
k = np.argmin(abs(w - wn))
ph = np.angle(Pxy[k]/Pxx[k])
print()
print("equivalent first-order lag at the pendulum frequency: %.2f s   (equivalent pure delay: %.2f s)"
      % (np.tan(-ph)/wn, -ph/wn))
if abs(Pxy[k])**2/(Pxx[k]*Pyy[k]) < 0.7:
    print("coherence is low at the pendulum frequency, so treat the numbers above as rough")
