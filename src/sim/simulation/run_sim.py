"""
uv run run_sim.py
uv run run_sim.py --arch drone_lqr --ref-for drone
uv run run_sim.py --arch payload_lqr --ref-for payload
"""

import argparse
import math

import numpy as np

import Prm.config as config
import sim.drone_test_only_mission_manager as mission
from sim import plotting
from sim.simulation import control_law
from sim.simulation.model import nonlinear_ode, payload_state

from sim.estimation.ekf import EKF, T_IB_fn


REF_TARGETS = ('payload', 'drone')


class _OffsetReference:
    """
    A reference shifted by a constant position offset
    """

    def __init__(self, ref, offset):
        self._ref = ref
        self._offset = np.asarray(offset, dtype=float)
        self.duration = ref.duration

    def __call__(self, t):
        p, v = self._ref(t)
        return p + self._offset, v


def as_payload_reference(ref, ref_target):
    """
    The payload reference the the controller
    """
    if ref_target == 'drone':
        return _OffsetReference(ref, [0, 0, -config.TETHER_LEN])
    return ref


def initial_state(pl_ref, pert=None):
    """
    Drone hovering at rest with the payload hanging on the t=0 reference
    """
    x0 = np.zeros(16)
    p_pl0, _ = pl_ref(0)
    x0[0:3] = p_pl0 + np.array([0, 0, config.TETHER_LEN])
    if pert is not None:
        x0 = x0 + pert
    return x0


def wind(acc=np.array([0, 1, 0]), transform=None):
    """
    Inputs an ENU vector of the wind acceleration in the inertial frame and accepts a transform for putting the wind in another reference frame.
    """
    acc = np.asarray(acc)

    if transform is None:
        return acc

    transformed_acc = transform @ acc

    return transformed_acc


def synthetic_measurement(x, T_IB, sigma_xy, sigma_yaw, rng, psi_p=0):
    """
    The bearing a camera would see, from truth, plus noise
    """
    alx, aly = x[12], x[13]
    q_I = np.array([math.sin(alx)*math.cos(aly),
                    math.sin(aly),
                    -math.cos(alx)*math.cos(aly)])

    b = config.CAM_R.T @ T_IB.T @ q_I

    z = np.array([b[0], b[1], psi_p]) + np.array([rng.normal(0, sigma_xy),
                                                  rng.normal(0, sigma_xy),
                                                 rng.normal(0, sigma_yaw)])

    return z


def simulate(architecture, ref, dt=1/config.CONTROL_FREQUENCY, ekf=False, ref_target='payload',
             cam_every=config.CONTROL_FREQUENCY // 30, wind_acc=None):

    if ref_target not in REF_TARGETS:
        raise ValueError(f'ref_target must be one of {REF_TARGETS}')

    pl_ref = as_payload_reference(ref, ref_target)
    ts = np.arange(0, ref.duration, dt)

    X = np.zeros((len(ts), 16))
    err_log = np.zeros((16, len(ts)))
    u_log = np.zeros((4, len(ts)))
    xi_log = np.zeros((5, len(ts)))
    var_log = np.zeros((5, len(ts)))

    pert = np.zeros(16)
    pert[12] = math.radians(30)
    pert[13] = math.radians(20)

    X[0] = initial_state(pl_ref, pert)
    filt = EKF(0, 0, 0, 0, 0, 0)
    rng = np.random.default_rng(0)
    a_prev = np.zeros(3)

    for i, t in enumerate(ts):
        x = X[i]

        T_IB = T_IB_fn(x[6], x[7], x[8])
        filt.xi, filt.P = filt.ekf_predict(filt.xi, filt.P, a_prev, dt)
        if i % cam_every == 0:
            z = synthetic_measurement(x, T_IB, filt.sigma_xy,
                                      filt.sigma_yaw, rng)
            filt.xi, filt.P = filt.update_with_z(filt.xi, filt.P, z, T_IB)
        xi_log[:, i] = filt.xi
        var_log[:, i] = np.diag(filt.P)

        x_hat = x
        if ekf:
            x_hat = x.copy()
            x_hat[12:16] = filt.xi[0:4]

        u, e = architecture(x_hat, t, pl_ref)
        xdot = nonlinear_ode(x, u)
        if wind_acc is not None:
            if i == 2000:
                wind_acc = [elem*2 for elem in wind_acc]

            xdot[3:6] += wind(wind_acc, T_IB)
        a_prev = xdot[3:6]

        u_log[:, i] = u
        err_log[:, i] = e

        if i + 1 < len(ts):
            X[i + 1] = x + dt*xdot

    P_ref = np.array([ref(t)[0] for t in ts])
    p_PL, v_PL = payload_state(X)
    p_tgt, v_tgt = ((X[:, 0:3], X[:, 3:6]) if ref_target == 'drone'
                    else (p_PL, v_PL))
    err_log[0:3] = (p_tgt - P_ref).T
    err_log[3:6] = (v_tgt - np.array([ref(t)[1] for t in ts])).T

    return ts, X, P_ref, err_log, u_log, xi_log, var_log


def main():
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument('--arch', default=control_law.DEFAULT_ARCHITECTURE,
                    choices=sorted(control_law.ARCHITECTURES),
                    help='control architecture from control_law.py')

    ap.add_argument('--ref-for', default='payload', choices=REF_TARGETS,
                    dest='ref_target',
                    help='which body the reference trajectory is drawn for')

    ap.add_argument('--ekf', action="store_true")
    ap.add_argument('--wind', default=[1, 0, 0])

    args = ap.parse_args()

    startPointHoverTime = 30
    endPointHoverTime = 30
    architecture = control_law.build(args.arch)

    ref = mission.ReferenceTrajectory(p_start=np.array([0, 0, 15]),
                                      p_end=np.array([0, -10, 15]),
                                      speed=1,
                                      startPointHoverTime=startPointHoverTime,
                                      endPointHoverTime=endPointHoverTime)

    ts, X, P_ref, err_log, u_log, xi_log, var_log = simulate(
        architecture, ref, ekf=args.ekf, ref_target=args.ref_target, wind_acc=args.wind)

    print(f'{args.arch} ({args.ref_target} reference): {len(ts)} samples '
          f'over {ts[-1]:.1f} s')

    plotting.plot_run(dict(ts=ts, X=X, P_ref=P_ref, err_log=err_log,
                           u_log=u_log, xi_log=xi_log, var_log=var_log,
                           arch=args.arch,
                           ref_target=args.ref_target),
                      layout="panels")


if __name__ == '__main__':
    main()
