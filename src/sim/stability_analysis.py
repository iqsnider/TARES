import numpy as np
import sim.dynamics as dyn
import matplotlib.pyplot as plt
import pandas as pd

STATE_LABELS = ["s1", "s2", "s3", "v1", "v2", "v3",
                "a1", "a2", "a1d", "a2d", "xI1", "xI2", "xI3"]
SWING_LABELS = ["a1", "a2", "a1d", "a2d"]

lqi = dyn.OuterLoopPayloadLQI()
A_cl = lqi.Abar - lqi.Bbar @ lqi.Kbar
poles, eigvecs = np.linalg.eig(A_cl)

left_eigvecs = np.linalg.inv(eigvecs)
participation = np.abs(eigvecs)*np.abs(left_eigvecs.T)
participation_factor = participation / participation.sum(axis=0)

part_df = pd.DataFrame(np.round(participation_factor, decimals=4),
                       index=STATE_LABELS)

for i, p in enumerate(poles):
    wn = abs(p)
    print(f"{p.real:+.4f} {p.imag:+.4f}j    wn={wn:.3f} rad/s   zeta={abs(p.real)/wn:.3f}")
    print(part_df.loc[STATE_LABELS, i].to_string())
    # plt.scatter(p.real, p.imag)

# plt.xlabel("Re")
# plt.ylabel("Im")
# plt.show()
