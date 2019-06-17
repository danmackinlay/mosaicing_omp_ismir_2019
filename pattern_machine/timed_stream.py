
"""
Timed-stream functions - objects that map times to values
"""
import numpy.random as rnd


def as_timed_stream(obj):
    """
    Turn scalars into timed_stream,
    and callables into iterables of invocation
    """
    if not callable(obj):
        return Const(obj)
    return obj


class TimedStream:
    def __repr__(self):
        return "{}(**{!r})".format(
            self.__class__.__name__,
            vars(self)
        )


class Const(TimedStream):
    def __init__(self, val):
        self.val = val

    def __call__(self, t, *args, **kwargs):
        return self.val


class Affine(TimedStream):
    def __init__(self, offset, mul):
        self.offset = offset
        self.mul = mul

    def __call__(self, t, *args, **kwargs):
        return self.mul * t + self.offset


class Fn(TimedStream):
    """Call a function that ignores the time argument
    """
    def __init__(self, func):
        self.func = func

    def __call__(self, t, *args, **kwargs):
        return self.func(*args, **kwargs)


class Gaussian(TimedStream):
    """Gaussian noise
    """
    def __init__(self, loc=0.0, scale=1.0, size=None):
        self.loc = loc
        self.scale = scale
        self.size = size

    def __call__(self, t, *args, **kwargs):
        return rnd.normal(low=self.loc, high=self.scale, size=self.size)


class UniformRand(TimedStream):
    """Uniform noise
    """
    def __init__(self, low=0.0, high=1.0, size=None):
        self.low = low
        self.high = high
        self.size = size

    def __call__(self, t, *args, **kwargs):
        return rnd.uniform(low=self.low, high=self.high, size=self.size)


class DictStream(TimedStream):
    def __init__(self, arg_dict):
        self._keys = arg_dict.keys()
        self._value_streams = [
            as_timed_stream(v) for v in arg_dict.values()
        ]

    def __call__(self, t, *args, **kwargs):
        return dict(zip(
            self._keys,
            [v(t, *args, **kwargs) for v in self._value_streams]
        ))