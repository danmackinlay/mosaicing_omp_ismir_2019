from scipy.stats import gamma
import numpy as np


def gamma_v(mean, sd, size=1):
    """
    simulate a gamma RV with desired mean and stddev
    scipy uses shape-scale (inverse rate)
    parameterisation with a side-order of scipy weirdness.
    (i.e. every dist has a location and scale param
    no matter how awkward or abnormal that makes it)

    mean = shape * scale
    sd = sqrt(shape) * scale
    i.e.
    shape = (mean/sd)^2
    scale = mean/shape
    """
    if sd <= 0 and size > 1:
        return np.full(size, mean)
    elif sd <= 0:
        return mean
    shape = (mean/sd)**2
    scale = mean / shape
    return gamma.rvs(shape, loc=0.0, scale=scale, size=size)


def gamma_v_prop(mean, prop, *args, **kwargs):
    """
    express sd as a fraction of mag
    """
    return gamma_v(mean, prop*mean)
