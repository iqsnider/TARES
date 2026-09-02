import numpy as np
import sim.dynamics as dyn
import matplotlib.pyplot as plt

import Prm.config as config

STATE_LABELS = ["s1", "s2", "s3", "v1", "v2", "v3",
                "a1", "a2", "a1d", "a2d", "xI1", "xI2", "xI3"]
SWING_LABELS = ["a1", "a2", "a1d", "a2d"]

lqi = dyn.OuterLoopPayloadLQI()
A_cl = lqi.Abar - lqi.Bbar @ lqi.Kbar
poles, eigvecs = np.linalg.eig(A_cl)


def step_response(step_distance=1):
    """
    Returns lean angle and commanded acceleration
    angle, acc
    """
    acc = lqi.K[0, 0]*step_distance
    angle = np.degrees(np.arctan(acc/config.GRAVITY))

    return angle, acc


for i, p in enumerate(poles):
    wn = abs(p)
    print(f"{p.real:+.4f} {p.imag:+.4f}j    wn={wn:.3f} rad/s   zeta={abs(p.real)/wn:.3f}")

period = 2*np.pi*np.sqrt(config.TETHER_LEN/config.GRAVITY)
angle, acc = step_response(0.5)

print(f"\n1 m payload step   acc={acc:.2f} m/s2   lean={angle:.1f} deg")
print(f"\n{config.AIRFRAME}   L={config.TETHER_LEN} m   "
      f"pendulum period={period:.2f} s")

plt.scatter(poles.real, poles.imag, marker="x", color="blue")
plt.axhline(0, color="gray", lw=0.5)
plt.axvline(0, color="gray", lw=0.5)
plt.xlabel("Re [1/s]")
plt.ylabel("Im [rad/s]")
plt.title(f"{config.AIRFRAME} payload LQI closed-loop poles")
plt.grid(True)
plt.show()


lqr = dyn.OuterLoopPayloadLQR()
A_cl = lqr.A - lqr.B @ lqr.subK
poles, eigvecs = np.linalg.eig(A_cl)


def step_response(step_distance=1):
    """
    Returns lean angle and commanded acceleration
    angle, acc
    """
    acc = lqr.K[0, 0]*step_distance
    angle = np.degrees(np.arctan(acc/config.GRAVITY))

    return angle, acc


for i, p in enumerate(poles):
    wn = abs(p)
    print(f"{p.real:+.4f} {p.imag:+.4f}j    wn={wn:.3f} rad/s   zeta={abs(p.real)/wn:.3f}")

period = 2*np.pi*np.sqrt(config.TETHER_LEN/config.GRAVITY)
angle, acc = step_response(0.5)

print(f"\n1 m payload step   acc={acc:.2f} m/s2   lean={angle:.1f} deg")
print(f"\n{config.AIRFRAME}   L={config.TETHER_LEN} m   "
      f"pendulum period={period:.2f} s")

plt.scatter(poles.real, poles.imag, marker="x", color="blue")
plt.axhline(0, color="gray", lw=0.5)
plt.axvline(0, color="gray", lw=0.5)
plt.xlabel("Re [1/s]")
plt.ylabel("Im [rad/s]")
plt.title(f"{config.AIRFRAME} payload LQR closed-loop poles")
plt.grid(True)
plt.show()
