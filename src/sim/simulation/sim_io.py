import os

import numpy as np

DEFAULT_RESULTS = os.path.join('out', 'sim_results.npz')

# keys stored in the archive, in the order the runner produces them
KEYS = ('ts', 'X', 'P_ref', 'err_log', 'u_log')


def save_results(path, ts, X, P_ref, err_log, u_log, arch='',
                 ref_target='payload'):
    """Write a simulation run to `path` (.npz). Returns the path.

    `ref_target` records which body `P_ref` (and the first six rows of
    `err_log`) belong to: 'payload' or 'drone'.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    np.savez(path, ts=ts, X=X, P_ref=P_ref, err_log=err_log, u_log=u_log,
             arch=arch, ref_target=ref_target)
    return path


def load_results(path):
    """Read a run written by `save_results`. Returns a dict."""
    with np.load(path) as data:
        out = {k: data[k] for k in KEYS}
        out['arch'] = str(data['arch'])
        # runs saved before ref_target existed always tracked the payload
        out['ref_target'] = (str(data['ref_target'])
                             if 'ref_target' in data else 'payload')
    return out
