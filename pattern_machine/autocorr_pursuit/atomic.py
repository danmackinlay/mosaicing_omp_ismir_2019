from autograd import numpy as np
from scipy.optimize import minimize
from autograd import grad
from autograd.builtins import dict
from numpy.linalg import lstsq, LinAlgError, norm
from random import random
from math import pi
import librosa
import warnings
from ..pseudorandom import prr
from .subatomic import (
    decaycos_eval,
    decaycos_int,
    decaycos_product,
    decaycos_self_product,
    decaycos_self_product2
)

"""
The main functions here are the matching pursuits for atomic decompositions of
a particular spectrogram frame into decaying sinusoid atoms.

The functions in this operate on arrays
and return tuples of arrays and scalars.
"""


def atoms_eval(t, mag, w, tau, phi=0.0, **kwargs):
    """
    We include a magnitude.
    """
    return mag * decaycos_eval(
        t,
        w, tau, phi, **kwargs
    )


def atoms_int(mag, w, tau, phi=0.0, L=128.0, **kwargs):
    return decaycos_int(
        w, tau, phi, L=L, **kwargs
    ) * mag


def atoms_product(
        mag, mag1,
        w, w1,
        tau, tau1,
        phi=0.0, phi1=0.0,
        L=128.0, **kwargs):
    return 0.5 * (
        mag * mag1 * decaycos_product(
            w, w1,
            tau, tau1,
            phi, phi1,
            L=L, **kwargs
        )
    )


def atoms_self_product(mag, w, tau, phi=0.0, L=128.0, **kwargs):
    return np.square(mag) * (
        decaycos_self_product(w, tau, phi, L=L, **kwargs)
    )


def atoms_self_product2(mag, w, tau, phi=0.0, L=128.0, **kwargs):
    return np.square(mag) * (
        decaycos_self_product2(w, tau, phi, L=L, **kwargs)
    )


def squashed_params(mag, w, tau, phi=None):
    """
    rescaled parameterization of decaycos_eval is more stable for optimization
    """
    return mag, w, squashed_alpha(tau), phi


def unsquashed_params(mag, w, tau, phi=None):
    """
    rescaled parameterization of decaycos_eval is more stable for optimization
    """
    return mag, w, unsquashed_alpha(tau), phi


def squashed_alpha(rate):
    """
    """
    return rate * 1e-2


def unsquashed_alpha(tau):
    """
    """
    return tau * 1e2


def atom_x0(
        target,
        n_starts=1,
        min_rand=0.3,
        max_rand=1.7,
        a=7.3, c=19.7,
        **kwargs):
    """
    Prior for atom params given a vector target
    """
    x0 = np.zeros((4, n_starts))
    w = guess_w(target, **kwargs)
    x0[0] = np.std(target) * np.sign(target[0])
    x0[1] = np.fmod(
        w * prr(
            n=n_starts,
            low=np.linspace(1, min_rand, n_starts, endpoint=True),
            high=np.linspace(1, max_rand, n_starts, endpoint=True),
            a=a,
            c=c
        ),
        np.pi
    )
    return x0


def guess_w(target, **kwargs):
    return (
        max(librosa.zero_crossings(target).sum() - 0.5, 1)  # bias down
    ) / target.size * np.pi


def choose_atom(
        target,
        maxiter=150,
        method='TNC',  # 'TNC', 'SLSQP', 'L-BFGS-B'
        t=None,
        opt_args={},
        n_starts=5,
        pursuit_a=7.3,
        pursuit_c=19.7,
        verbose=0,
        **atom_args,):

    # We want to select atoms
    # by maximising inner product.
    if t is None:
        t = np.arange(target.size)
    else:
        t = np.asarray(t)

    biggest = np.max(np.abs(target))

    def multi_objective(params):
        return np.square(
            target.reshape(-1, 1) -
            atoms_eval(
                np.reshape(t, (-1, 1)),
                *squashed_params(*(params.reshape(4, -1)))
            ),
        ).sum(0)

    def local_objective(params):
        # mag, w, tau, phi = params
        return multi_objective(params).sum(0)

    # def callback_fun(x):
    #     print('x', x)
    #     plt.figure()
    #     plt.plot(target, label='targ')
    #     plt.plot(decaycos_eval(t, *x), label='atom')
    #     plt.legend()
    #     plt.show()

    x0 = atom_x0(
        target,
        n_starts=n_starts,
        a=pursuit_a,
        c=pursuit_c,
        **atom_args)
    x0_unsq = unsquashed_params(*x0)
    x0_unsq = np.array(x0_unsq).ravel()
    # print('x0', x0)

    jac = grad(local_objective)

    bounds = [
        (-2 * biggest, 2 * biggest),  # mag
        (0.0, pi),  # w
        (-2.0, 10.0),  # unsquashed tau
        (None, None),  # phi
     ] * n_starts

    res = minimize(
        local_objective,
        x0_unsq,
        method=method,
        jac=jac,
        bounds=bounds,
        # callback=callback_fun,
        options=dict(
            maxiter=maxiter,
            disp=verbose >= 10,
            gtol=1e-5,  # def 1e-5
            # ftol=1e-11,  # def 2.22-09
            eps=1e-08,  # def 1e-8
            **opt_args))
    # loss = res.fun
    losses = multi_objective(res.x)
    best = np.argmin(losses)
    # import pdb; pdb.set_trace()
    x_unsq = res.x.reshape(4, -1)[:, best]
    return squashed_params(*x_unsq)


def decaycos_atomic_pursuit_precond(
        target,
        *args, **kwargs):

    # scale all problems to be similar size wrt optimizer tolerance
    scale = np.std(target) + 1e-16
    coef, loss = decaycos_atomic_pursuit(target/scale, *args, **kwargs)
    coef[0] *= scale
    return coef, loss * scale, scale


def decaycos_atomic_pursuit(
            target,
            basis_size=8,  # >=2
            maxiter=150,
            rtol=0.01,
            rcond=1e-3,   # lax colinearity condition for approximation
            cutoff=20.0,  # higher mags than this are unlikely to be stable
            verbose=0,
            stop_early=True,
            **atom_args):
    """
    The atom matching pursuit problem is where we try to approximate a given
    correlation profile with decaying sinusoid atoms.
    """

    n_pts = target.size
    t = np.arange(n_pts)

    residual = target

    x = np.zeros((4, basis_size))
    basis_eval = np.zeros((n_pts, basis_size))

    bias = np.mean(target)
    x[:, 0] = [bias, 0.0, 0.0, 0.0]  # bias
    basis_eval[:, 0] = 1  # bias

    residual -= bias
    loss = np.sqrt(np.mean(np.square(residual)))
    last_i = 1

    for i in range(1, basis_size):
        if np.std(residual) < rtol and stop_early:
            break
        # progress = (i-1) / (basis_size-2)
        atom_coef = choose_atom(
            residual,
            t=t,
            **atom_args,
        )
        mag, w, tau, phi = atom_coef
        # print('prm', mag, w, tau, phi)
        x[:, i] = atom_coef
        # atom_eval = decaycos_eval(t, w, tau, phi).ravel()
        # residual = target - atom_eval
        basis_eval[:, i] = decaycos_eval(
            t.reshape(-1, 1),
            w, tau, phi
        ).squeeze(1)
        try:
            new_mags, sum_resid, rank, s = lstsq(
                basis_eval[:, :i+1],
                target.reshape(-1, 1),
                rcond=rcond)
        except LinAlgError as e:
            if verbose >= 4:
                warnings.warn(
                    'exploding solution at step {}\n'
                    '{}'.format(
                        i, e
                    )
                )
            mag, w, tau, phi = 0.0, random()*pi, 0.0, 0.0
            basis_eval[:, i] = decaycos_eval(t, w, tau, phi)
            x[:, i] = mag, w, tau, phi

        if np.max(new_mags) > cutoff and verbose >= 4:
            # exploding results indicate colinear atoms;
            # randomize and null
            warnings.warn(
                'exploding solution at step {}\n'
                'try raising `rcond`:\n'
                '{}'.format(
                    i, new_mags
                )
            )
            mag, w, tau, phi = 0.0, random()*pi, 0.0, 0.0
            basis_eval[:, i] = decaycos_eval(t, w, tau, phi)
            x[:, i] = mag, w, tau, phi
        else:
            x[0, :i + 1] = new_mags.ravel()

        curr_approx = np.dot(
            basis_eval[:, :i+1],
            x[0, :i+1].reshape(-1, 1)
        ).ravel()

        new_residual = target - curr_approx
        # plt.figure()
        # plt.plot(target, label='target')
        # plt.plot(curr_approx, label='hat')
        # plt.plot(residual, label='oldres')
        # plt.plot(new_residual, label='newres')
        # plt.plot(basis_eval[:, i]*mag, label='new_atom')
        # plt.legend()
        # plt.show()
        # import pdb; pdb.set_trace()
        residual = new_residual
        loss = np.sqrt(np.mean(np.square(residual)))
        last_i = i

    x[3] = np.fmod(x[3], np.pi)
    loss = np.sqrt(np.mean(np.square(residual)))
    if stop_early:
        return x[:, :last_i], loss
    else:
        return x, loss


def decaycos_ts_pursuit_raw(
        corr_array,
        basis_size=5,
        **kwargs):

    # correlation is packed timewise;
    # (time_step, period)
    # unlike librosa spectrograms, which are
    # (period, time_step)
    # but there is much less mess if we have time as the 1st axis
    # in the approximand
    n_t = len(corr_array)

    approx_coef_all = np.zeros((n_t, 4, basis_size))
    losses = np.zeros((n_t,))

    for step, target in enumerate(corr_array):
        coef, local_loss, scale = decaycos_atomic_pursuit_precond(
            target,
            basis_size=basis_size,
            # x0=f,
            **kwargs)
        approx_coef_all[step] = coef
        losses[step] = local_loss
        # if res.fun>0.01:
        #     raise Exception
        step += 1

    return approx_coef_all, losses
