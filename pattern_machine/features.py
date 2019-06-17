import numpy as np
from pprint import pformat


class Struct:
    def __init__(
            self,
            sr=44100,
            **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

    def __repr__(self):
        return "{}(**{})".format(
            self.__class__.__name__,
            pformat(vars(self))
        )


class SampleStruct(Struct):
    def __init__(
            self,
            sr=44100,
            **kwargs):
        super().__init__(**kwargs)
        self.sr = sr


class FrameFeature(SampleStruct):
    """
    feature based on a single timeslice of a sample
    """
    def __init__(
            self,
            feature,
            frame_length=2048,
            start_samp=0,
            sr=44100,
            **kwargs):
        super().__init__(
            sr=sr,
            **kwargs
        )
        self.feature = feature
        self.start_samp = start_samp
        self.frame_length = frame_length


class SequenceFeature(SampleStruct):
    """
    features based on regular time slices of a sample.

    the storage of correlation bases is transposed with respect to librosa,
    which is to say, per default we iterate over time.
    Here is a convenient accessor.
    """
    def __init__(
            self,
            features,
            hop_length=512,
            frame_length=2048,
            start_samp=0,
            sr=44100,
            **kwargs):
        super().__init__(
            sr=sr,
            **kwargs
        )
        self.features = features
        self.hop_length = hop_length
        self.start_samp = start_samp
        self.frame_length = frame_length

    def at_samp(self, i):
        return self.frameat(self.samp2frame(i))

    def at_time(self, t):
        return self.frameat(self.time2frame(t))

    def asidx(self, frame_i):
        """
        clip some arbitrary float into a chacheable, in-bounds int
        """
        return np.int_(np.round(frame_i).clip(0, len(self.features)-1))

    def frameat(self, frame_i):
        return self[self.asidx(frame_i)]

    def __getitem__(self, frame_i):
        """
        feature extraction method, but without fancy range checking,
        multi-lookup etc
        """
        return self.features[frame_i]

    def frame2samp(self, frame_i):
        return (frame_i * self.hop_length) + self.start_samp

    def samp2frame(self, samp):
        return (samp - self.start_samp) / self.hop_length

    def frame2time(self, frame_i):
        return self.samp2time(self.frame2samp(frame_i))

    def time2frame(self, time):
        return self.samp2frame(self.time2samp(time))

    def samp2time(self, samp):
        return samp / self.sr

    def time2samp(self, time):
        return time * self.sr

    def __iter__(self):
        for feature in self.features:
            yield feature

    def __len__(self):
        return len(self.features)

    def pairs(self):
        for i in range(len(self)):
            yield self.frame2time(i), self[i]

    def times(self):
        for i in range(len(self)):
            yield self.frame2time(i)

    def first_time(self):
        return self.start_samp/self.sr

    def last_time(self):
        return self.frame2time(len(self))
