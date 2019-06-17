"""
Audio mangling. Sample-rate-blind, operates on raw arrays
"""
import numpy as np
from librosa.core import resample
from librosa.filters import get_window as librosa_get_window
from .jcache import memory


def transpose(audio_data, rate=1.0, mul=1.0, **kwargs):
    """
    see https://librosa.github.io/librosa/generated/librosa.core.resample.html
    """
    return resample(
        y=audio_data,
        orig_sr=44100.0,  # dummy value for resampler
        target_sr=44100.0 / rate,
        **kwargs
    ) * mul


@memory.cache
def get_window(window='hann', n_samples=2048,):
    """
    get a tapering window. Length in samples.
    """
    return librosa_get_window(window, n_samples, fftbins=True)


def windowed(audio_data, window='hann'):
    """
    return a windowed version of a sample
    """
    env = get_window(window, audio_data.size)
    return audio_data * env


def taper(audio_data, n_samples=512, window='hann',  **window_args):
    """
    taper just the ends of a sample.
    """
    env = get_window(window=window, n_samples=n_samples * 2, **window_args)
    gain = 1.0/ np.max(env)  # should be 1 in the middle now
    audio_data = np.copy(audio_data)
    audio_data[:n_samples] *= env[:n_samples] * gain
    audio_data[-n_samples:] *= env[n_samples:] * gain
    return audio_data
