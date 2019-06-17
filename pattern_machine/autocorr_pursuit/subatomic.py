from autograd import numpy as np

"""
evaluate and integrate decaying sinusoid atoms.

The functions in this operate on arrays
and return tuples of arrays and scalars.
"""


def decaycos_eval(t, w, tau, phi=None, **kwargs):
    """
    evaluate the decaycos thing at each time.
    """
    if phi is None:
        phi = np.zeros_like(w)

    return (
        np.cos(
            t * w +
            phi
        ) * np.exp(-t * tau)
    )


def decaycos_int(w, tau, phi=0.0, L=128.0, verbose=0, **kwargs):
    r"""
    integral from 0 to L.

    Using Euler identities we find the antiderivative

    $$\begin{aligned}
    \int^L\cos (\omega \xi)\exp (-\tau \xi)\dd \xi
    &=  \frac{
            e^{-L \tau } (
                \omega  \sin (L \omega +\phi )-
                \tau  \cos (L \omega +\phi )
            )
        }{\tau ^2+\omega ^2}
    \end{aligned}$$
    and thus

    $$\begin{aligned}
    \int_0^L\cos (\omega \xi )\exp (-\tau \xi )\dd \xi
    &=\frac{1}{\tau ^2+\omega ^2}\left.
        e^{- \xi  \tau } (\omega  \sin ( \xi  \omega +\phi )-\tau  \cos ( \xi  \omega +\phi ))
    \right|_{ \xi =0}^{ \xi =L}\\
    &=\frac{1}{\tau ^2+\omega ^2}\left(
        e^{-L \tau } (
            \omega  \sin ( L  \omega +\phi )
            -\tau  \cos ( L  \omega +\phi )
        )
        -\omega  \sin \phi + \tau  \cos \phi )
    \right).\\
    \end{aligned}$$
    """
    comps = (
        np.exp(-L*tau) * (
            -tau * np.cos(L * w + phi)
            + w * np.sin(L * w + phi)
        ) + (
            tau * np.cos(phi)
            - w * np.sin(phi)
        )
    )/(np.square(tau) + np.square(w))
    # This approximation is numerically stable as tau and w go to 0,
    # and autograd can  (usually) differentiate it correctly;
    # there are issues with the gradient if tau is 0 but no other terms are
    explode_mask = np.isfinite(comps) < 1
    comps = np.nan_to_num(comps) + L * explode_mask
    return comps


def decaycos_product(
        w, w1,
        tau, tau1,
        phi=0.0, phi1=0.0,
        L=128.0, **kwargs):
    r"""$$
    \inner{\cos (\omega \xi + \phi) \exp -\tau \xi}{\cos (\omega' \xi' + \phi') \exp -\tau' \xi}_v&= \begin{array}{l}
    \frac{1}{2} \int_0^L\cos(\omega_{-} \xi + \phi_{-} ) \exp (-\tau_{+} \xi) \dd \xi \\+
    \frac{1}{2} \int_0^L \cos(\omega_{+} \xi + \phi_{+} ) \exp (-\tau_{+} \xi) \dd \xi
    \end{array}\\
    &= \begin{array}{l}
    \frac{1}{2}  \left.
        \frac{e^{-\xi \tau_{+}  } \left(\omega_{-} \sin (\xi \omega_{-} +\phi_{-} )- \tau_{+} \cos (\xi \omega_{-} + \phi_{-} )\right)}{\tau_{+}^2+\omega_{-}^2}
    \right|_{\xi=0}^{\xi=L}
    \\ + \frac{1}{2}  \left.
        \frac{e^{-\xi \tau_{+}  } \left(\omega_{+} \sin (\xi \omega_{+} +\phi_{+} )- \tau_{+} \cos (\xi \omega_{+} + \phi_{+} )\right)}{\tau_{+}^2+\omega_{+}^2}
    \right|_{\xi=0}^{\xi=L}
    \end{array}\\
    &= \begin{array}{l}
    \frac{1}{2(\tau_{+}^2+\omega_{-}^2)} \left(
        e^{-L \tau_{+}  } \left(
            \omega_{-} \sin (L \omega_{-} +\phi_{-} )-
            \tau_{+} \cos (L \omega_{-} + \phi_{-} )
        \right) -
        \omega_{-} \sin \phi_{-} +
        \tau_{+} \cos \phi_{-}
    \right)
    \\+ \frac{1}{2(\tau_{+}^2+\omega_{+}^2)} \left(
        e^{-L \tau_{+}  } \left(
            \omega_{+} \sin (L \omega_{+} +\phi_{+} )-
            \tau_{+} \cos (L \omega_{+} + \phi_{+} )
        \right) -
        \omega_{+}\sin \phi_{+} +
        \tau_{+}\cos \phi_{+}
    \right).
    \end{array}\\
    \end{aligned}
    $$
    """
    return 0.5 * (
        decaycos_int(
            w - w1, tau + tau1, phi - phi1, L=L, **kwargs
        ) + decaycos_int(
            w + w1, tau + tau1, phi + phi1, L=L, **kwargs
        )
    )


def decaycos_self_product(w, tau, phi=0.0, L=128.0, **kwargs):
    r"""
    Squared integral
    $$
    \|\cos (\omega \xi + \phi) \exp \tau \xi\|_v^2
        =   \frac{1}{2} \int_0^L e^{-2 \xi \tau} \cos(2 \xi \omega +2 \phi ) \dd \xi +
            \frac{1}{2} \int_0^L e^{-2 \xi \tau}\dd \xi
    $$
    and
    $$
    \frac{1}{2}\int_0^L e^{-2 \xi \tau}\dd \xi
        = \frac{1-e^{-2L\tau}}{4 \tau}
    $$
    """
    return (
        0.5 * decaycos_int(
            w * 2.0,
            tau * 2.0,
            phi * 2.0,
            L=L,
            **kwargs
        ) - 0.25 * np.expm1(-2.0 * L * tau)/tau
    )


def decaycos_self_product2(w, tau, phi=0.0, L=128.0, **kwargs):
    r"""
    Squared integral, unoptimized.
    For cross-checking optimized version.
    """
    return decaycos_product(
        w, w,
        tau, tau,
        phi, phi,
        L=L,
    )
