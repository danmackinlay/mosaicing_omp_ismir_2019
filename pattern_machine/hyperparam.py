from itertools import product
from random import sample
import numpy as np
from librosa.util import normalize
from .autocorr_pursuit.molecular import (
    choose_molecule_pitch_opt
)
from .autocorr import autocorr_t
from .autocorr_pursuit.main import GrainCodebook
from .sample import Grain


def make_some_codes(samples, n_grain_per_samp=64):
    codes = GrainCodebook()
    for s in samples:
        ts = np.random.random(n_grain_per_samp) * s.duration()
        for t in ts:
            codes.append(Grain(s, t))
    return codes


def gd_eval(codes, grain_is, lr=0.05, maxiter=20, verbose=0, **kwargs):
    code_dot = []
    for i, (src_i, tgt_i) in enumerate(grain_is):
        print("{}/{}: {}, {}". format(i+1, len(grain_is), src_i, tgt_i))
        target = normalize(autocorr_t(
            codes.sample(tgt_i),
            codes.t(tgt_i),
        ).feature, norm=2)
        trajectory = choose_molecule_pitch_opt(
            target,
            codes.acorr_coef(src_i),
            trace=True,
            lr=lr,
            maxiter=maxiter,
            verbose=verbose,
            **kwargs
        )
        code_dot.append(
            trajectory
        )
    return code_dot


def hyperparam_search_lr(
        codes,
        lrs=10**np.linspace(-5, 0, 11, endpoint=True),
        maxiter=20,
        n_samp=100,
        verbose=True,
        **kwargs):
    parm_dot = {}
    grain_is = sample(
        list(product(range(len(codes)), range(len(codes)))),
        n_samp)

    for lr in lrs:
        if verbose >= 0:
            print("lr", lr)
        parm_dot[lr] = gd_eval(
            codes, grain_is,
            lr=lr,
            maxiter=maxiter,
            verbose=verbose,
            **kwargs)

    return parm_dot


def hyperparam_search_n_start(
        codes,
        n_starts=np.arange(1, 64, 4),
        maxiter=20,
        n_samp=100,
        verbose=True,
        **kwargs):
    parm_dot = {}
    grain_is = sample(
        list(product(range(len(codes)), range(len(codes)))),
        n_samp)

    for n_s in n_starts:
        if verbose >= 0:
            print("n_s", n_s)
        parm_dot[n_s] = gd_eval(
            codes, grain_is, n_starts=n_s,
            maxiter=maxiter,
            verbose=verbose,
            **kwargs)

    return parm_dot


def extract_traj(trajectories, idx=0):
    all_obj = [[h[idx] for h in g] for g in trajectories]
    all_obj = np.moveaxis(np.array(all_obj), 2, 1)
    all_obj = all_obj.reshape((-1, all_obj.shape[-1]))
    return all_obj.T


def extract_max_traj(trajectories, idx=0):
    all_obj = [[h[idx] for h in g] for g in trajectories]
    # from IPython.core.debugger import set_trace; set_trace()
    best_obj = np.amax(np.array(all_obj), 2)
    best_obj = best_obj.reshape((-1, best_obj.shape[-1]))
    return best_obj.T


def quantilify(X):
    return np.percentile(X, [0, 5, 25, 50, 75, 95, 100], axis=1)
