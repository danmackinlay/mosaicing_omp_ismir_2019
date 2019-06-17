"""
Analysis of Sample objects sound files into spectral-peak and RMS features
"""

import librosa
from .sample import sample_from
from .jcache import memory


@memory.cache
def highlights(
        sample, top_db=30,
        frame_length=2048,
        hop_length=512,
        align_zeros=True):
    edges = librosa.effects.split(
        sample.get_all_audio_data(),
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    new_y = librosa.effects.remix(
        sample.get_all_audio_data(), edges,
        align_zeros=align_zeros)

    return sample_from(
        sample,
        y=new_y,
        append_stem="_highlights"
    )
