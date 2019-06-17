"""
Harmonic percussive separation
"""

import librosa
import librosa.cache
from .jcache import memory


@memory.cache
def hpss(sample, persist=True):
    # Separate into harmonic and percussive. I think this preserves phase?
    H, P = librosa.decompose.hpss(sample.D)
    # Resynthesize the harmonic component as waveforms
    y_harmonic = librosa.util.fix_length(
        librosa.istft(
            H,
            # window=window,
        ),
        len(sample.get_all_audio_data())
    )
    harmonic_sample = sample_from(
        sample,
        y=y_harmonic,
        D=H,
        append_stem="_harmonic",
    )
    if persist:
        harmonic_sample.save()

    y_percussive = librosa.util.fix_length(
        librosa.istft(
            P,
            # window=window,
        ),
        len(sample.get_all_audio_data())
    )
    percussive_sample = sample_from(
        sample,
        y=y_percussive,
        D=P,
        append_stem="_percussive",
    )
    if persist:
        percussive_sample.save()

    return (
        harmonic_sample,
        percussive_sample
    )
