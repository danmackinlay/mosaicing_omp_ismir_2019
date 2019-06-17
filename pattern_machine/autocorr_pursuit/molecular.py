from autograd import numpy as np
from autograd import elementwise_grad
from numpy.linalg import lstsq, LinAlgError
import warnings
from .subatomic import decaycos_int
# The following are needed in both atomic and molecular contexts
from .atomic import (
    atoms_eval,
    atoms_int as molecular_int,
    atoms_product,
    atoms_self_product
)

"""
The main functions here are the molecular
decomposition of a spectrogram in terms of atoms.

The functions in this module operate on arrays and return tuples of
arrays and scalars.
"""


def molecular_eval(t, mag, w, tau, phi=0.0, **kwargs):
    """
    Nearly the same as for atoms, but we no longer broadcast generally.
    t is always the first index, which is time.
    Other params are a given molecule.
    """
    t = np.reshape(t, (-1, 1))
    return atoms_eval(t, mag, w, tau, phi=0.0, **kwargs).sum(1)


def molecular_eval_norm(
        t, mag, w, tau, phi=None, L=128.0,
        norm_method='analytic',
        verbose=0,
        **kwargs):
    """
    Nearly the same as for atoms, but we no longer broadcast generally.
    One molecule, many time steps.
    """
    # verbose = kwargs.get('verbose', 0)
    mol_norm = molecular_mag(
        t, mag, w, tau, phi=phi, L=L,
        norm_method=norm_method, verbose=verbose,
        **kwargs)
    if np.any(mol_norm <= 0.0) and verbose > 1:
        # Null molecules should be removed before we get to this stage
        # but occasionally my logic misses them
        warnings.warn(
            "magnitude of mol norm vanished {}".format(
               (mag, w, tau, phi)
            )
        )
        mol_norm = np.maximum(mol_norm, 1e-12)

    mol_ev = molecular_eval(
        t,
        mag, w, tau, phi=None, L=128.0,
        **kwargs
    )
    if not np.all(np.isfinite(mol_ev)) and verbose >= 1:
        warnings.warn(
            "raw molecule exploded {} giving \n{}".format(
                [mag, w, tau, phi], mol_ev,
            )
        )

    normed_eval = mol_ev/mol_norm

    if not np.all(np.isfinite(normed_eval)):
        warnings.warn(
            "raw molecule exploded {} giving \n{}".format(
                [mag, w, tau, phi], normed_eval,
            )
        )

    return normed_eval


def molecular_mag(
        t, mag, w, tau, phi=None, L=128.0, norm_method='analytic',
        **kwargs):
    # verbose = kwargs.get('verbose', 0)
    if norm_method == 'analytic':
        magsq = molecular_norm2_analytic(
            mag, w, tau, phi=phi, L=L, **kwargs)
        # if (not np.isreal(magsq)) or (not np.isfinite(magsq)) or magsq < 0.0:
        #     warnings.warn("invalid sqmagnitude {} for {}".format(
        #         magsq,
        #         [mag, w, tau, phi]
        #     ))
        #     from IPython.core.debugger import set_trace; set_trace()

        return np.sqrt(magsq)
    else:
        return np.sqrt(molecular_norm2_empirical(
            t, mag, w, tau, phi=phi,))


def molecular_norm2_empirical(
        t, mag, w, tau, phi=None, **kwargs):
    """
    Approximate method for calculating molecule norm.
    Provided for cross-checking.
    """
    return np.square(molecular_eval(
        t,
        mag, w, tau, phi=None, L=128.0,
        **kwargs
    )).sum(0)


def molecular_norm2_analytic_comp(
        mag, w, tau, phi=None, L=128.0,
        verbose=0, clip=True, **kwargs):
    """
    Squared definite integral of the molecule from 0 to L.
    Could be optimized by exploiting symmetry over diagonal.
    """
    if phi is None:
        phi = np.zeros_like(mag)
    mag2 = mag.reshape(1, -1) * mag.reshape(-1, 1)
    wp = w.reshape(1, -1) + w.reshape(-1, 1)
    phip = phi.reshape(1, -1) + phi.reshape(-1, 1)
    wm = w.reshape(1, -1) - w.reshape(-1, 1)
    phim = phi.reshape(1, -1) - phi.reshape(-1, 1)
    taup = tau.reshape(1, -1) + tau.reshape(-1, 1)

    return mag2 * 0.5 * (
        decaycos_int(wp, taup, phip, L=L, verbose=verbose)
        + decaycos_int(wm, taup, phim, L=L, verbose=verbose)
    )


def molecular_norm2_analytic(
        mag, w, tau, phi=None, L=128.0,
        verbose=0,
        # clip=True,
        **kwargs):
    """
    Squared definite integral of the molecule from 0 to L.
    Could be optimized by exploiting symmetry over diagonal.
    """
    # if phi is None:
    #     phi = np.zeros_like(mag)
    # mag2 = mag.reshape(1, -1) * mag.reshape(-1, 1)
    # wp = w.reshape(1, -1) + w.reshape(-1, 1)
    # phip = phi.reshape(1, -1) + phi.reshape(-1, 1)
    # wm = w.reshape(1, -1) - w.reshape(-1, 1)
    # phim = phi.reshape(1, -1) - phi.reshape(-1, 1)
    # taup = tau.reshape(1, -1) + tau.reshape(-1, 1)

    # sqmag = (mag2 * (
    #     decaycos_int(wp, taup, phip, L=L)
    #     + decaycos_int(wm, taup, phim, L=L)
    # )).sum()
    sqmag = molecular_norm2_analytic_comp(
        mag, w, tau, phi=phi,
        L=L,
        verbose=verbose,
        **kwargs).sum()
    if sqmag < 0.0 and verbose > 5:
        warnings.warn("squared magnitude {} negative with mag {}\nw {}\ntau {}\nphi {}".format(
            sqmag,
            mag,
            w,
            tau,
            phi
        ))
    # if clip:
    #     return np.maximum(sqmag, 0.0)
    return sqmag


def molecular_prod(coef1, coef2, L=128.0, verbose=0):
    """
    non-normalised analytic molecular product.
    """
    mag1, w1, alpha1, phi1 = coef1
    mag2, w2, alpha2, phi2 = coef2
    mag12 = mag1.reshape(1, -1) * mag2.reshape(-1, 1)
    w1p2 = w1.reshape(1, -1) + w2.reshape(-1, 1)
    phi1p2 = phi1.reshape(1, -1) + phi2.reshape(-1, 1)
    w1m2 = w1.reshape(1, -1) - w2.reshape(-1, 1)
    phi1m2 = phi1.reshape(1, -1) - phi2.reshape(-1, 1)
    alpha12 = alpha1.reshape(1, -1) + alpha2.reshape(-1, 1)

    return (mag12 * 0.5 * (
        decaycos_int(w1p2, alpha12, phi1p2, L=L, verbose=verbose)
        + decaycos_int(w1m2, alpha12, phi1m2, L=L, verbose=verbose)
    )).sum()


def molecular_prod_norm_int(coef1, coef2, L=128.0):
    """
    product of two analytic molecules, normalised wrt the first.
    Delete me.
    """
    mag1, w1, alpha1, phi1 = coef1
    scale = molecular_mag(*coef1)
    return molecular_prod(
        (mag1/scale, w1, alpha1, phi1),
        coef2,
        L=L
    )


def molecular_diff_l2(coef1, coef2, L=128.0, **kwargs):
    mag1, w1, alpha1, phi1 = coef1
    mag2, f2, alpha2, phi2 = coef2

    S1 = molecular_prod(coef1, coef1, L=L)
    S2 = molecular_prod(coef2, coef2, L=L)
    S12 = molecular_prod(coef1, coef2, L=L)
    return S1 - 2 * S12 + S2


def molecular_diff_lp(coef1, coef2, L=128.0, t=None, p=1):
    """
    approximate $L_p$ distance over a grid t
    """
    if t is None:
        t = np.linspace(
            0,
            L + 1,
            # endpoint=True
        )

    S1 = molecular_eval(t, *coef1)
    S2 = molecular_eval(t, *coef2)
    return np.mean(np.abs(S2-S1))**p * L


def molecular_diff_linf(coef1, coef2, L=128.0, t=None, p=1):
    """
    approximate $L_{\infty}$ distance over a grid t
    """
    if t is None:
        t = np.linspace(
            0,
            L + 1,
            # endpoint=True
        )

    S1 = molecular_eval(t, *coef1)
    S2 = molecular_eval(t, *coef2)
    return np.max(np.abs(S2 - S1)) * L


def molecular_compose(coefs, gains, rates):
    """
    return many molecules as a single scaled molecule
    """
    alphas = []
    mags = []
    ws = []
    phis = []
    for (mag, w, tau, phi), gain, rate in zip(coefs, gains, rates):
        mags.append(mag * gain)
        ws.append(w * rate)
        alphas.append(tau * rate)
        phis.append(phi)
    return np.stack((
        np.concatenate(mags),
        np.concatenate(ws),
        np.concatenate(alphas),
        np.concatenate(phis)
    ))


def molecular_scale(coefs, gain, rate):
    """
    scale a single molecule
    """
    (mag, w, tau, phi) = coefs
    return np.array([
        mag.ravel() * gain,
        w.ravel() * rate,
        tau.ravel() * rate,
        phi.ravel()
    ])


def choose_molecule(
        target,
        code_coefs,
        pitched=True,
        verbose=0,
        **kwargs):
    if pitched:
        return choose_molecule_pitched(
            target,
            code_coefs,
            verbose=verbose,
            **kwargs)
    else:
        return choose_molecule_fixed(
            target,
            code_coefs,
            verbose=verbose,
            **kwargs)


def choose_molecule_fixed(
        target,
        code_coefs,
        t=None,
        **molecule_args):
    raise NotImplementedError()


# @memory.cache
def choose_molecule_pitch_opt(
        target,
        code_coef,
        maxiter=5,
        t=None,
        lr=0.01,
        low_pitch=0.5**0.5,
        high_pitch=2.0**0.5,
        n_starts=65,
        trace=False,
        pdb=False,
        verbose=0,
        norm_method='analytic',
        **molecule_args):
    """
    choose pitch for one molecule and return inner product at that pitch
    """
    if t is None:
        t = np.arange(target.size)

    rates = np.exp(np.linspace(
        np.log(low_pitch),
        np.log(high_pitch),
        n_starts+2,
        endpoint=True
    )[1:-1])

    max_step = (high_pitch-low_pitch)/n_starts

    def multi_objective(rates):
        """
        normalised inner product for each rate
        """
        molecules = [
            molecular_scale(
                code_coef,
                1,
                rate,
            ) for rate in rates
        ]
        normecules = np.array([
            molecular_eval_norm(
                t,
                *molecule,
                norm_method=norm_method,
                verbose=verbose)
            for molecule
            in molecules
        ])
        if not np.all(np.isfinite(normecules)) and verbose >= 1:
            exploded = np.isfinite(normecules.sum(1))
            warnings.warn(
                "{} normed molecules {} exploded with\n{} at rates\n{}".format(
                    np.sum(exploded),
                    normecules.shape,
                    code_coef,
                    rates[exploded]
                )
            )
        obj = np.array([
            np.dot(
                normecule,
                target
            )
            for normecule in normecules
        ])
        return np.nan_to_num(obj)

    grad = elementwise_grad(multi_objective)

    # f, axarr = plt.subplots(2, 1)
    if trace:
        trace_list = []
    for step_i in range(maxiter):
        # gradient ascent
        jac = grad(rates)
        if not np.all(np.isfinite(jac)) and verbose >= 1:
            warnings.warn(
                "jac exploded {}\nfor coefs \n{}\nat rate {}\nwith obj {}".format(
                    jac,
                    code_coef,
                    rates,
                    multi_objective(rates)
                )
            )
        jac = np.nan_to_num(jac)
        step = np.clip(lr*jac, -max_step, max_step)
        if pdb:
            print(
                step_i,
                "jac", np.sqrt((jac**2).mean()),
                "step", np.sqrt((step**2).mean()))
            from IPython.core.debugger import set_trace; set_trace()

        # val = multi_objective(rates)
        # best = np.argmax(val)
        # stepsize = jac[best]
        # print('stepsize', stepsize)

        # axarr[0].quiver(
        #     rates,  # X
        #     val,  # Y
        #     step,  # U
        #     np.zeros_like(step),  # V
        #     np.full_like(step, step_i/(maxiter-1)),  # C
        #     cmap="magma",
        #     angles='xy',
        #     label="step {}".format(step_i))
        # axarr[1].scatter(
        #     rates,  # X
        #     val,  # Y
        #     cmap="magma",
        #     label="step {}".format(step_i))

        rates = rates + step  # gradient ascent step
        rates = np.clip(rates, low_pitch, high_pitch)
        if trace:
            trace_list.append(
                (multi_objective(rates), rates, jac, step)
            )
        if verbose >= 21:
            max_goodness = np.amax(multi_objective(rates))
            print("max_goodness at ", step_i,)
            if not np.isfinite(max_goodness):
                from IPython.core.debugger import set_trace; set_trace()

    if trace:
        return trace_list

    goodnesses = multi_objective(rates)
    best_idx = np.argmax(goodnesses)

    if verbose >= 11:
        print(
            "choose_molecule_pitch_opt",
            best_idx,
            rates[best_idx],
            "@",
            goodnesses[best_idx],
        )

    return rates[best_idx], goodnesses[best_idx]


def choose_molecule_pitched(
        target,
        code_coefs,
        t=None,
        verbose=0,
        match_rtol=0.01,
        **molecule_args):

    if t is None:
        t = np.arange(target.size)

    best_rate = 1
    best_idx = -1
    best_goodness = -1e9  # or -inf?
    if verbose >= 15:
        print("=== ")
    if verbose >= 25:
        print("coeffs\n", code_coefs)
    for idx, coef in enumerate(code_coefs):
        if verbose >= 15:
            print("finding goodness for", idx, coef)
        rate, goodness = choose_molecule_pitch_opt(
            target,
            coef,
            t=t,
            verbose=verbose,
            **molecule_args
        )
        if verbose >= 15:
            print("found goodness for", idx, goodness, rate)
        if goodness > best_goodness:
            best_goodness = goodness
            best_rate = rate
            best_idx = idx

    if verbose >= 11:
        print(
            "choose_molecule_pitched",
            best_idx,
            best_rate
        )

    return best_rate, best_idx


def molecular_pursuit(
        target,
        code_coefs,
        basis_size=4,
        rtol=0.01,        # Dependence on this parameter is sensitive
        rcond=1e-3,       # lax colinearity condition for approximation
        cutoff=100.0,     # higher gains than this are unlikely to be stable
        pitched=True,
        match_rtol=0.01,  # stop search early if good enough
        verbose=0,
        **molecule_args):
    """
    In the molecular matching pursuit problem we approximate a target
    correlation profile with a codebook of correlation molecules by
    maximising inner product.

    There are various distinctions between this and the atomic case.

    * we need to preserve the identity of the molecules
    * we don't need to precondition the tau term since tau and w are now coupled
    * so we only have 2 coefs
    * no bias term
    * tedious to get a basis dictionary that has unit gain
    * despite this we normalise and do a true matching pursuit
    * ...
    """
    n_pts = target.size
    t = np.arange(n_pts)

    scale = max(np.sqrt(np.sum(np.square(target))), 1e-8)
    init_scale = scale
    deviance = scale
    # print("scale", scale)
    residual = target

    gain_rate = np.zeros((2, basis_size))
    gain_rate[1] = 1
    molecule_idx = -np.ones(basis_size, dtype=int)

    basis_eval = np.zeros((n_pts, basis_size))

    for i in range(basis_size):
        # print("-----", i)
        # progress = (i-1) / (basis_size-2)
        rate, idx = choose_molecule(
            residual,
            code_coefs,
            t=t,
            pitched=pitched,
            verbose=verbose,
            match_rtol=match_rtol,
            **molecule_args,
        )
        molecule_idx[i] = idx
        # print('prm', mag, w, tau, phi)
        gain_rate[:, i] = rate
        # molecule_eval = decaycos_eval(t, 1, w, tau, phi).ravel()
        # residual = target - molecule_eval
        basis_eval[:, i] = molecular_eval(
            t, *molecular_scale(code_coefs[idx], 1.0, rate)
        )
        gain_scale, sum_resid, rank, s = lstsq(
            basis_eval[:, :i + 1],
            target.reshape(-1, 1),
            rcond=rcond)
        # print(basis_eval[:, :i + 1])
        # print("gain scale\n", gain_scale.ravel())
        # print("gain rate\n", gain_rate[:, :i + 1])

        if (
                (
                    np.max(np.abs(
                        gain_rate[0, :i + 1] *
                        gain_scale.ravel()
                    )) > cutoff
                ) and verbose >= 14):
            # exploding results indicate colinear molecules;
            warnings.warn(
                'exploding solution at step {}\n'
                'try raising `rcond`:\n'
                '{}->{}'.format(
                    i,
                    gain_scale.ravel(),
                    gain_rate[0, :i + 1].ravel() * gain_scale.ravel()
                )
            )
            # Now what?
        else:
            gain_rate[0, :i + 1] *= gain_scale.ravel()
            basis_eval[:, :i + 1] *= gain_scale.reshape((1, -1))

        curr_approx = np.sum(
            basis_eval[:, :i + 1],
            axis=1
        )

        new_residual = target - curr_approx
        # plt.figure()
        # plt.plot(target, label='target')
        # plt.plot(curr_approx, label='hat')
        # plt.plot(residual, label='oldres')
        # plt.plot(new_residual, label='newres')
        # plt.plot(basis_eval[:, i]*mag, label='new_molecular')
        # plt.legend()
        # plt.show()
        # import pdb; pdb.set_trace()
        residual = new_residual
        new_deviance = np.sqrt(np.sum(np.square(new_residual)))
        # print("deviance", deviance, "/", scale)
        if deviance - new_deviance < rtol * scale:
            # we didn't improve the match so we won't next step either
            if verbose >= 17:
                print(
                    "failed to improve",
                    new_deviance, "-", deviance,
                    "<", rtol, "*", scale
                )
            break

        deviance = new_deviance
        scale /= new_deviance

    loss = deviance/scale
    return gain_rate[:, :i+1], molecule_idx[:i+1], loss, init_scale
