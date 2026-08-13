import sim.dynamics as dyn

import numpy as np

def get_outer_loop_controller_eigvals(A,B,K):
    eigs = np.linalg.eigvals(A - B@K)
    return eigs

lqi = dyn.OuterLoopPayloadLQI()

for rho in np.logspace(-2,2,20):
    K = lqi._lqr(lqi.Abar, lqi.Bbar, lqi.Q, rho*lqi.R)
    print(rho, np.sort(get_outer_loop_controller_eigvals(lqi.Abar, lqi.Bbar, K).real))

