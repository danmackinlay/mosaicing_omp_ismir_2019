"""
sound file structs
"""
from math import ceil, floor
from pathlib import Path

import numpy as np
import librosa
from librosa.util import frame
from .features import SequenceFeature, SampleStruct
from . import sfio
from .jcache import strid, hash_content
from .audio_edit import transpose, windowed, taper
from .math import clip
import dotenv
import os


class Time:
    def __init__(self, val, unit='s', sr=44100):
        self.val = val
        self.unit = unit
        self.sr = sr

    def as_n_samples(self, sr=44100):
        if self.unit == 's':
            return ceil(self.val * (self.sr or sr))
        elif self.unit == 'samp':
            return self.val

    def as_seconds(self, sr=44100):
        if self.unit == 's':
            return self.val
        elif self.unit == 'samp':
            return self.val / (self.sr or sr)


class Sample(SequenceFeature):
    def __init__(
            self,
            y,
            sr=44100,
            stem="sample",
            qual="",
            suffix=".mp3",
            parent_path=".",
            hop_length=512,
            frame_length=2048,
            duration=None,
            **kwargs):
        end = y.shape[-1]
        if duration is not None:
            end = min(duration * sr, end)
        end = int(end)
        self.end = end
        self._y = y
        features = frame(
            y,
            hop_length=hop_length,
            frame_length=frame_length
        )
        super().__init__(
            features=features,
            hop_length=hop_length,
            frame_length=frame_length,
            sr=sr,
            **kwargs)
        for k in kwargs:
            setattr(self, k, kwargs[k])
        self.stem = stem
        self.qual = qual
        self.suffix = suffix
        self.parent_path = Path(parent_path)
        self.start_samp = 0
        self.end_samp = y.shape[-1]

    def __hash__(self):
        return hash_content(memoryview(self.get_all_audio_data()))

    def __repr__(self):
        return "{}({!r},sr={},duration={})".format(
            self.__class__.__name__,
            self.path(),
            self.sr,
            self.duration()
        )

    def _repr_html_(self):
        import IPython.display as ipd
        widget = ipd.Audio(self.get_all_audio_data(), rate=self.sr)
        return widget._repr_html_()

    def duration(self):
        return (self.end_samp - self.start_samp)/self.sr

    def strid(self):
        return strid(memoryview(self.get_all_audio_data()))

    def update_path(self, file_path):
        if file_path is None:
            self.stem = ""
            self.qual = "|" + strid(memoryview(self._y))
            self.suffix = ".aiff"
            self.parent_path = Path("")
        else:
            file_path = Path(file_path)
            self.stem = file_path.stem
            self.parent_path = file_path.parent
            self.qual = ""
            self.suffix = str(file_path.suffix)

    def set_default_dir(self):
        dotenv.load_dotenv(dotenv.find_dotenv())
        self.parent_path = Path(os.environ.get("PS_SAMPLE_OUTPUT_DIR", ""))

    def path(self):
        return str(self.parent_path / (self.stem + self.qual + self.suffix))

    def normalize(self, norm=np.inf, ):
        self._y = librosa.util.normalize(self._y, norm=norm)

    def taper(self, *args, **kwargs):
        self._y = taper(self._y, *args, **kwargs)

    def get_all_audio_data(
            self,
            **kwargs):
        return self.get_audio_data_samp(
            0, self.end, **kwargs)

    def get_audio_data_t(
            self,
            location=0.0,
            duration=None,
            **kwargs):
        if duration is not None:
            duration = int(duration*self.sr)
        return self.get_audio_data_samp(
            int(location*self.sr), duration, **kwargs)

    def get_audio_data_samp(
            self,
            location=0,
            duration=None,
            anchor='left',
            **kwargs):
        if anchor == 'left':
            return self.get_audio_data_left_samp(
                location=location, duration=duration, **kwargs)
        elif anchor.startswith('cent'):
            return self.get_audio_data_centered_samp(
                location=location, duration=duration, **kwargs)
        else:
            raise ValueError('unknown {}'.format(anchor))

    def get_audio_data_left_samp(
            self, location=0, duration=None, **kwargs):
        location = int(location)
        duration = (duration or self.frame_length)
        notional_left = floor(location)
        notional_right = floor(location+duration)
        left = clip(notional_left, 0, self._y.shape[-1])
        right = clip(notional_right, 0, self._y.shape[-1])
        left_pad = left - notional_left
        right_pad = notional_right - right
        y = self._y[location:location+duration]
        if left_pad == 0 and right_pad == 0:
            return y
        return np.pad(y, ((int(left_pad), int(right_pad))), 'constant')

    def get_audio_data_centered_samp(
            self, location=0.0, duration=None, **kwargs):
        duration = (duration or self.frame_length)
        notional_left = floor(location-duration*0.5)
        notional_right = floor(location+duration*0.5)
        left = clip(notional_left, 0, self._y.shape[-1])
        right = clip(notional_right, 0, self._y.shape[-1])
        left_pad = left - notional_left
        right_pad = notional_right - right
        # print(
        #     "get_audio_data_centered_samp",
        #     center, length, left, right, left_pad, right_pad)
        y = self._y[int(left):int(right)]
        if left_pad == 0 and right_pad == 0:
            return y
        return np.pad(y, ((int(left_pad), int(right_pad))), 'constant')

    def set_audio_data_t(self, audio_data, location=0.0, **kwargs):
        return self.set_audio_data_samp(
            audio_data, int(location*self.sr), **kwargs)

    def set_audio_data_samp(
            self,
            audio_data,
            location=0,
            anchor='left',
            **kwargs):
        if anchor == 'left':
            return self.set_audio_data_left_samp(
                audio_data, location=location, **kwargs)
        elif anchor.startswith('cent'):
            return self.set_audio_data_centered_samp(
                audio_data, location=location, **kwargs)
        else:
            raise ValueError('unknown {}'.format(anchor))

    def set_audio_data_left_samp(
            self,
            audio_data,
            location=0,
            anchor='left',
            **kwargs):
        start = int(clip(location, 0, self.end))
        end = int(clip(audio_data.shape[-1]+location, 0, self.end))
        self._y[start:end] = audio_data[:end-location]

    def set_audio_data_centered_samp(self, audio_data, location, **kwargs):
        """
        Centred writes have complicated logic because of remaining
        centered at boundaries.
        """
        that_length = audio_data.shape[-1]
        this_length = self._y.shape[-1]
        nominal_this_left = floor(location-that_length/2)
        nominal_this_right = floor(location+that_length/2)
        this_left = clip(nominal_this_left, 0, this_length)
        this_right = clip(nominal_this_right, 0, this_length)
        that_left = this_left - nominal_this_left
        that_right = that_length - nominal_this_right + this_right
        # print(
        #     "set_audio_data_centered_samp",
        #     "at", location,
        #     "that length", that_length,
        #     "this_length", this_length,
        #     "thises", this_left, this_right, this_right - this_left,
        #     "thats", that_left, that_right, that_right - that_left)
        self._y[this_left:this_right] = audio_data[that_left:that_right]

    def overdub_audio_data_t(self, audio_data, start=0.0, **kwargs):
        return self.overdub_audio_data_samp(
            audio_data, int(start * self.sr), **kwargs)

    def overdub_audio_data_samp(
            self,
            audio_data,
            location=0,
            anchor='left',
            **kwargs):
        if anchor == 'left':
            return self.overdub_audio_data_left_samp(
                audio_data,
                location=location,
                **kwargs)
        elif anchor.startswith('cent'):
            return self.overdub_audio_data_centered_samp(
                audio_data,
                location=location,
                **kwargs)
        else:
            raise ValueError('unknown {}'.format(anchor))

    def overdub_audio_data_left_samp(self, audio_data, location=0, **kwargs):
        return self.set_audio_data_left_samp(
            self.get_audio_data_left_samp(
                location, duration=audio_data.shape[-1]
            ) + audio_data,
            location=location,
            **kwargs)

    def overdub_audio_data_centered_samp(self, audio_data, location, **kwargs):
        """
        Centred overdubs have complicated logic because of remaining
        centered at boundaries.
        """
        return self.set_audio_data_centered_samp(
            self.get_audio_data_centered_samp(
                location, duration=audio_data.shape[-1]
            ) + audio_data,
            location=location,
            **kwargs)

    def save(self, file_path=None, format="s16le"):
        if file_path is not None:
            self.update_path(file_path)
        return sfio.save(
            self.path(),
            self.get_all_audio_data(),
            sr=self.sr,
            format=format,
            norm=False)

    def qualify_name(self, chunk):
        if len(chunk):
            self.qual = "|" + chunk
        else:
            self.qual = ""

    def append_stem(self, chunk):
        self.stem += chunk

    def mutate_name(self):
        self.qualify_name(self.strid())

    def quantize_time_to_samp(
            self, t, quantum=None, **kwargs):
        quantum = quantum or self.hop_length
        return quantize_time_to_samp(
            t,
            quantum=quantum,
            start_samp=self.start_samp,
            sr=self.sr,
            **kwargs)


def sample_from(
        sample,
        append_stem=None,
        suffix=".aiff",
        **kwargs):
    defaults = dict(
        sr=sample.sr,
        stem=sample.stem,
        suffix=sample.suffix,
        parent_path=sample.parent_path,
    )
    defaults.update(kwargs)
    sample = Sample(**defaults)
    # sample.mutate_name()
    if append_stem is not None:
        sample.append_stem(append_stem)
    return sample


def load_sample(
        file_path,
        nchan=1,
        offset=0.0,
        duration=120.0,
        n_fft=4096,
        window='hann',
        hop_length=1024,
        sr=44100,
        **kwargs):
    y, sr = sfio.load(
        str(file_path),
        offset=offset,
        nchan=nchan,
        duration=duration,
        sr=sr,
        **kwargs
    )
    D = librosa.stft(
        y,
        n_fft=n_fft,
        # window=window,
        hop_length=hop_length
    )
    s = Sample(
        y=y,
        D=D,
        sr=sr,
        nchan=nchan,
        n_fft=n_fft,
        hop_length=hop_length,
        file_path=file_path,
        **kwargs
    )
    s.update_path(file_path)
    return s


def zero_sample(
        duration=5.0,
        sr=44100,
        **kwargs):

    return Sample(
        y=np.zeros(int(duration*sr)),
        sr=sr,
        duration=duration,
        **kwargs)


class Grain(SampleStruct):
    """
    Centred ref to a sub-Sample
    """
    def __init__(
            self,
            sample,
            t,
            length=4096,  # default
            anchor='center',
            **kwargs):
        super().__init__(
            **kwargs)
        self.sample = sample
        self.t = t
        self.length = length
        self.anchor = anchor

    def __hash__(self):
        return hash((self.sample, self.t, self.anchor))

    def get_audio_data_samp(self, length=None, anchor=None):
        return self.sample.get_audio_data_samp(
            location=self.t * self.sample.sr,
            duration=length or self.length,
            anchor=anchor or self.anchor
        )

    def get_audio_data_t(self, duration=None, anchor=None):
        return self.sample.get_audio_data_t(
            location=self.t,
            duration=duration or self.length / self.sr,
            anchor=anchor or self.anchor
        )

    def duration(self):
        return self.length/self.sample.sr

    def avg_power(self, *args, **kwargs):
        data = self.get_audio_data_samp(*args, **kwargs)
        return np.square(data).mean()


def quantize_time_to_samp(
        t,
        quantum=4096,
        sr=44100,
        start_samp=0,
        **kwargs):
    """
    return a sample index on the sequence grid, which is cacheable.
    """
    return np.int(np.round((
        t*sr - start_samp
    ) / quantum, 0)) * quantum + start_samp


def overdub_t(
        dest_sample,
        source_sample,
        dest_t=None,
        source_t=None,
        duration=None,  # in source_t
        rate=1.0,
        mul=1.0,
        window='hann',
        source_anchor='left',
        dest_anchor='left',
        verbose=0,
        *args, **kwargs):
    # # We could round rate to 1/256, i.e. 8 bit fixed point,
    # # to cache it. Does this need caching?
    # rate = round(
    #     256 * rate * (dest_sample.sr / source_sample.sr)
    #     ) * 0.00390625
    source_data = source_sample.get_audio_data_t(
        source_t,
        anchor=source_anchor,
        duration=duration,
        verbose=verbose
    )
    if verbose > 8:
        print(
            "rendering t {}->{}".format(
                source_t,
                dest_t)
        )

    source_data_transposed = transpose(
        source_data,
        rate=rate,
        mul=mul,
    )
    if verbose > 12:
        print(
            "duration {} {} {} {}".format(
                duration,
                rate,
                source_data.size,
                source_data_transposed.size)
        )

    source_data_windowed = windowed(
        source_data_transposed,
        window=window,
        **kwargs
    )

    dest_sample.overdub_audio_data_t(
        source_data_windowed,
        dest_t,
        anchor=dest_anchor,
        verbose=verbose
    )
    return dest_sample
