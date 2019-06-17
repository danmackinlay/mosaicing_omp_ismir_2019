"""
Analysis of Sample objects sound files into spectral-peak and RMS features
"""

import numpy as np
from librosa.util import frame
from math import floor, sqrt
from .features import SequenceFeature, FrameFeature
from .jcache import memory
from .sample import quantize_time_to_samp
from librosa.filters import get_window as librosa_get_window


class AutoCorrSample(SequenceFeature):
    pass


class AutoCorrFrame(FrameFeature):
    """
    A single match
    """
    pass


@memory.cache
def get_half_corr_window(window='cosine', n_samples=2048):
    """
    get a tapering window. Length in samples.
    norm of window squared is 1.
    """
    if window in ('const', 'constant', None):
        # like a boxcar window, a flat window. But we can return a scalar
        return np.ones(n_samples) / sqrt(n_samples)
    w = librosa_get_window(window, n_samples, fftbins=True)
    power = np.square(w).sum()
    return w/sqrt(power)


@memory.cache
def get_corr_window(window='hann', n_samples=2048):
    """
    get a tapering window. Length in samples.
    L1 norm of window is 1.
    """
    if window in ('const', 'constant', None):
        # like a boxcar window, a flat window. But we can return a scalar
        return np.ones(n_samples) / n_samples
    w = librosa_get_window(window, n_samples, fftbins=True)
    energy = np.abs(w).sum()
    return w/energy


@memory.cache
def autocorr_featurize(
            sample,
            hop_length=1024,
            frame_length=4096,
            delay_step_size=8,
            n_delays=128,
            window='const',
            window_interleave=False
        ):
    """
    TODO: check for mono
    """
    y = sample.get_all_audio_data()
    # TODO: better highpass based on frame length
    y = y - np.mean(y)
    delays = np.arange(n_delays) * delay_step_size
    max_delay = np.max(delays)
    start_center_sample = (max_delay + frame_length) / 2
    n_frames = 1 + floor((y.size-max_delay-frame_length)/hop_length)
    # valid_samps = (n_frames - 1) * hop_length + frame_length
    del_frames_l = [
        frame(
            y[max_delay-d:y.size-d],
            hop_length=hop_length,
            frame_length=frame_length
        ) for d in delays
    ]
    now_frames = del_frames_l[0]
    feat_array = np.zeros((n_frames, n_delays))
    w = get_corr_window(window, n_samples=frame_length)
    w /= np.sqrt(np.sum(w**2))
    if window_interleave:
        ws = np.sqrt(w)

    for d_i in range(n_delays):
        if window_interleave:
            w = ws[0:frame_length-d] * ws[d:frame_length]

        np.sum(
            now_frames * del_frames_l[d_i] * w.reshape(-1, 1),
            axis=0,
            out=feat_array[:, d_i]
        )

    return AutoCorrSample(
        features=feat_array,
        start_sample=start_center_sample,
        sr=sample.sr,
        hop_length=hop_length,
        frame_length=frame_length,
        delay_step_size=delay_step_size,
        n_delays=n_delays
    )

@memory.cache
def autocorr_slice_featurize(
            sample,
            hop_length=1024,
            frame_length=4096,
            delay_step_size=8,
            n_delays=128,
            window='const',
            window_interleave=False
        ):
    """
    TODO: check for mono
    """
    y = sample.get_all_audio_data()
    # TODO: better highpass based on frame length
    y = y - np.mean(y)
    delays = np.arange(n_delays) * delay_step_size
    max_delay = np.max(delays)
    start_center_sample = (max_delay + frame_length) / 2
    n_frames = 1 + floor((y.size-max_delay-frame_length)/hop_length)
    # valid_samps = (n_frames - 1) * hop_length + frame_length
    w = get_corr_window(window, n_samples=frame_length)
    if window_interleave:
        ws = np.sqrt(w)

    feat_array = np.zeros((n_frames, n_delays))
    for f in range(n_frames):
        if window_interleave:
            w = ws[0:frame_length-d] * ws[d:frame_length]
        hop = f * hop_length
        now_frame = y[max_delay+hop:max_delay+hop+frame_length]
        now_frame = now_frame - np.mean(now_frame)
        now_frame = now_frame * w
        for d_i in range(n_delays):
            d = delays[d_i]
            feat_array[f, d_i] = np.sum(
                now_frame[0:frame_length-d] * now_frame[d:frame_length]
            )

    return AutoCorrSample(
        features=feat_array,
        start_sample=start_center_sample,
        sr=sample.sr,
        hop_length=hop_length,
        frame_length=frame_length,
        delay_step_size=delay_step_size,
        n_delays=n_delays
    )


def autocorr(
            y,
            samp,
            frame_length=4096,
            delay_step_size=8,
            n_delays=128,
            window='const',
            allow_short=True,
            window_interleave=False
        ):
    """autocorrelation of a bare array at a given sample"""
    w = get_corr_window(window, n_samples=frame_length)
    if window_interleave:
        ws = np.sqrt(w)
    # from IPython.core.debugger import set_trace; set_trace()
    max_delay = int((n_delays - 1) * delay_step_size)
    sub_y_len = int(max_delay + frame_length)
    notional_start_samp = int(max(samp - sub_y_len / 2, 0))
    notional_end_samp = int(notional_start_samp + sub_y_len)
    start_samp = max(notional_start_samp, 0)
    end_samp = min(notional_end_samp, y.size)
    if end_samp - start_samp < sub_y_len:
        if not allow_short:
            raise IndexError('sample too short')
        # I think I could do this re-padding without copying
        # by just being wise with indices.
        start_pad = start_samp - notional_start_samp
        end_pad = notional_end_samp - end_samp
        sub_y = np.pad(
            y[start_samp:end_samp],
            ((start_pad, end_pad)),
            'constant')
    else:
        sub_y = y[start_samp:end_samp]

    corr = np.zeros(n_delays, dtype=np.float32)
    len_sub_y = sub_y.size
    sub_y = sub_y - sub_y.mean()

    for d_i in range(n_delays):
        d = d_i * delay_step_size
        if window_interleave:
            w = ws[0:frame_length-d] * ws[d:frame_length]
        now = w * (sub_y[len_sub_y-frame_length:len_sub_y])[:w.size]
        corr[d_i] = np.dot(
            now,
            (sub_y[len_sub_y-frame_length-d:len_sub_y-d])[:w.size],
        ) / frame_length
    return corr


@memory.cache
def autocorr_samp(
            sample,
            samp,
            frame_length=4096,
            delay_step_size=8,
            n_delays=128,
            window='cosine',
            window_interleave=False
        ):
    y = sample.get_all_audio_data()
    corr = autocorr(
        y,
        delay_step_size*floor(samp/delay_step_size),
        frame_length,
        delay_step_size,
        n_delays,
        window=window,
        window_interleave=window_interleave)

    return AutoCorrFrame(
        feature=corr,
        sr=sample.sr,
        frame_length=frame_length,
        delay_step_size=delay_step_size,
        n_delays=n_delays
    )


def autocorr_t(
            sample,
            t,
            delay_step_size=8,
            **kwargs
        ):
    return autocorr_samp(
        sample,
        quantize_time_to_samp(
            t,
            quantum=delay_step_size,
            **kwargs),
        delay_step_size=delay_step_size,
        **kwargs)
