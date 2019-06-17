from random import random
from math import pi
import numpy as np


def triangle_map(mu):
    return 2 * np.arcsin(
        np.sin(mu * np.pi * 0.5)
    ) / np.pi


def pr_mu(mu, a=0.0, c=1.0):
    return triangle_map((mu + a) * c)


def prn(n, a=0.0, c=1.0, endpoint=True):
    """
    pseudorandom numbers on [-1, 1]
    """
    return (
        pr_mu(np.linspace(-1, 1, n, endpoint=endpoint), a, c)
    )


def prr(n=1, low=0.0, high=1, a=0.0, c=1.0, endpoint=True):
    """
    pseudorandom things on a range
    """
    return (prn(n, a, c, endpoint=endpoint) * 0.5 + 0.5) * (high-low) + low
