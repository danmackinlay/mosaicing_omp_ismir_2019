from itertools import repeat
from math import inf, sqrt
from .random import gamma_v


def flat_if_array(v):
    if np.isscalar(v):
        return v
    else:
        return np.ravel(v)


def iterize_kwargs(verbose=False, **kwargs):
    """
    Turn kwargs of scalars and interables into kwargs of iterables

    >> > for d in iterize_kwargs(a=[1, 2, 3], b=[4, 5, 6, 7]):
    >> > print(d)
    {'a': 1, 'b': 4}
    {'a': 2, 'b': 5}
    {'a': 3, 'b': 6}

    """
    keys = kwargs.keys()
    values = kwargs.values()
    for v in zip(*[as_iterable(value) for value in values]):
        yield dict(zip(keys, v))


def as_iterable(obj):
    """
    Turn scalar into iterables, and iterables into themselves,
    and callables into iterables of invocation
    """
    try:
        return iter(obj)
    except TypeError:
        if callable(obj):
            return iter_caller(obj)
        return repeat(obj)


def iter_caller(fn):
    """
    turn a function into an iterable that invokes the function
    """
    while True:
        yield fn()


def astep(start, step):
    """
    a = astep(1, 0.1)
    for s in islice(iter(a), 10):
        print(s)
    """
    v = start
    while True:
        yield v
        v += step


def gamma_renewal_proc_scale(
        interval, var=0.0,
        low=0.0, high=inf,
        verbose=0):
    t = low
    sd = sqrt(var)
    t = t + gamma_v(
        interval,
        sd)
    inc = t

    if verbose >= 5:
        print('ssd0', interval, sd, t)
        print('tmin', t, interval)
    while t < high:
        if verbose >= 21:
            print('ssd {:.5f} {:.5f} {:.5f} {:.5f}'.format(
                interval, var, t, inc))
        yield t
        inc = gamma_v(
            interval,
            sd)
        t = t + inc
    if verbose >= 5:
        print('tmax', t, high)


def poisson_proc_scale(
        interval,
        low=0.0, high=inf,
        verbose=0):
    return gamma_renewal_proc_scale(
        interval,
        var=interval,
        low=low,
        high=high,
        verbose=verbose
    )
