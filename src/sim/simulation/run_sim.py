"""
uv run run_sim.py
uv run run_sim.py --arch drone_pd --ref-for drone

plotting
uv run plotting.py out/sim_results.npz
"""

import argparse
import math

import numpy as np
from scipy.integrate import solve_ivp

from model import nonlinear_ode, payload_state
import Prm.config as config
import sim.drone_test_only_mission_manager as mission
from sim.simulation import control_law
from sim.simulation.sim_io import DEFAULT_RESULTS, save_results


REF_TARGETS = ('payload', 'drone')


class _OffsetReference:
    """A reference shifted by a constant position offset."""

    def __init__(self, ref, offset):
        self._ref = ref
        self._offset = np.asarray(offset, dtype=float)
        self.duration = ref.duration

    def __call__(self, t):
        p, v = self._ref(t)
        return p + self._offset, v


def as_payload_reference(ref, ref_target):
    """The payload reference the control stack tracks.

    Every architecture in control_law.py consumes a payload reference, so a
    trajectory drawn for the drone is dropped one tether length to give the
    payload curve that hangs the drone on the caller's original path.
    """
    if ref_target == 'drone':
        return _OffsetReference(ref, [0.0, 0.0, -config.TETHER_LEN])
    return ref


def initial_state(pl_ref):
    """Drone hovering at rest with the payload hanging on the t=0 reference."""
    x0 = np.zeros(16)
    p_pl0, _ = pl_ref(0.0)
    x0[0:3] = p_pl0 + np.array([0.0, 0.0, config.TETHER_LEN])
    return x0


def simulate(architecture, ref, dt=0.02, ref_target='payload'):
    """Integrate the closed loop while tracking `ref`.

    `ref_target` names the body `ref` was drawn for ('payload' or 'drone');
    the returned P_ref and error rows 0:6 are reported against that body.

    Returns ts, X, P_ref, err_log (16 x N), u_log (4 x N).
    """
    if ref_target not in REF_TARGETS:
        raise ValueError(f'ref_target must be one of {REF_TARGETS}')

    pl_ref = as_payload_reference(ref, ref_target)
    t_end = ref.duration
    t_eval = np.arange(0, t_end, dt)

    def ode(t, x):
        return nonlinear_ode(x, architecture(x, t, pl_ref)[0])

    sol = solve_ivp(ode, [0.0, t_end], initial_state(pl_ref),
                    t_eval=t_eval, method='RK45')

    X = sol.y.T
    P_ref = np.array([ref(t)[0] for t in sol.t])
    p_PL, v_PL = payload_state(X)
    # the body the reference was drawn for is the one we score against
    p_tgt, v_tgt = ((X[:, 0:3], X[:, 3:6]) if ref_target == 'drone'
                    else (p_PL, v_PL))

    err_log = np.zeros((16, len(sol.t)))
    u_log = np.zeros((4, len(sol.t)))

    for i, t in enumerate(sol.t):
        u_log[:, i], err_log[:, i] = architecture(X[i], t, pl_ref)
        err_log[0:3, i] = p_tgt[i] - P_ref[i]
        err_log[3:6, i] = v_tgt[i] - ref(t)[1]

    return sol.t, X, P_ref, err_log, u_log



def main():
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument('--arch', default=control_law.DEFAULT_ARCHITECTURE,
                    choices=sorted(control_law.ARCHITECTURES),
                    help='control architecture from control_law.py')

    ap.add_argument('--ref-for', default='payload', choices=REF_TARGETS,
                    dest='ref_target',
                    help='which body the reference trajectory is drawn for')

    ap.add_argument('-o', '--out', default=DEFAULT_RESULTS,
                    help='output .npz path')

    args = ap.parse_args()

    startPointHoverTime = 5
    endPointHoverTime = 5 
    architecture = control_law.build(args.arch)

    ref = mission.ReferenceTrajectory(p_start=np.array([0, 0, 15]),
                                      p_end=np.array([0, -10, 15]),
                                      speed=1,
                                      startPointHoverTime=startPointHoverTime,
                                      endPointHoverTime=endPointHoverTime)

    ts, X, P_ref, err_log, u_log = simulate(architecture, ref,
                                            ref_target=args.ref_target)

    path = save_results(args.out, ts, X, P_ref, err_log, u_log, arch=args.arch,
                        ref_target=args.ref_target)
    print(f'{args.arch} ({args.ref_target} reference): {len(ts)} samples '
          f'over {ts[-1]:.1f} s -> {path}')


if __name__ == '__main__':
    main()
