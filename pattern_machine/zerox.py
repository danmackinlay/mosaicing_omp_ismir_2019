"""
Zero crossings,
inspired by
https://github.com/algorithmic-music-exploration/amen/blob/master/amen/timing.py#L45
"""

import librosa
from ..sample import sample_from
import librosa.cache
from .jcache import memory

from bisect import bisect_left, bisect_right


@memory.cache
def zero_indexes(sample):
    """
    Create zero crossing indexes.
    We use these in synthesis, and it is easier to make them here.
    """
    zero_indexes = []
    for channel_index in range(sample.num_channels):
        channel = sample.get_all_audio_data()[channel_index]
        zero_crossings = librosa.zero_crossings(channel)
        zero_index = np.nonzero(zero_crossings)[0]
        zero_indexes.append(zero_index)
    return zero_indexes


@memory.cache
def _get_offsets(sample, starting_sample, ending_sample, num_channels):
    """
    Find the offset to the next zero-crossing, for each channel.
    """
    offsets = []
    for zero_index in zero_indexes(sample):
        index = bisect_left(zero_index, starting_sample) - 1
        if index < 0:
            starting_offset = 0
        else:
            starting_crossing = zero_index[index]
            starting_offset = starting_crossing - starting_sample

        index = bisect_left(zero_index, ending_sample)
        if index >= len(zero_index):
            ending_offset = 0
        else:
            zci = min(bisect_right(zero_index, ending_sample), len(zero_index) - 1)
            ending_crossing = zero_index[zci]
            ending_offset = ending_crossing - ending_sample

        offsets.append((starting_offset, ending_offset))

    if num_channels == 1:
        results = (offsets[0], offsets[0])
    elif num_channels == 2:
        results = (offsets[0], offsets[1])

    return results
