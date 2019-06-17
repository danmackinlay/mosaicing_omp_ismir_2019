import numpy as np
from scipy.signal import filtfilt
import resampy

from .filter_design import RC


def filt_f(wavdata, sr=22050.0, f=20.0, btype='highpass', **kwargs):
    """
    remove the bottom few Hz (def 20Hz)
    """
    w = 0.5 * f/float(sr)
    b, a = RC(Wn=w, btype=btype)
    # NB filtfilt seems to return fortan-contiguous arrays somehow?
    return np.ascontiguousarray(filtfilt(b, a, wavdata, **kwargs))


def filt_w(wavdata, w=0.01, btype='highpass', **kwargs):
    """
    remove by angular freq
    """
    b, a = RC(Wn=w, btype=btype)
    # NB filtfilt seems to return fortan-contiguous arrays somehow?
    return np.ascontiguousarray(filtfilt(b, a, wavdata, **kwargs))


def normalized(wavdata, norm=1):
    """
    Centered, normalized array.
    """
    wavdata -= wavdata.mean()
    wavdata *= 1.0/(np.abs(wavdata)**norm).max()
    return wavdata
