from autograd import numpy as np
from ..features import SequenceFeature, SampleStruct
from ..autocorr import autocorr_samp, autocorr_t, autocorr_featurize
from ..jcache import memory
from ..pseudorandom import prr
from ..sample import quantize_time_to_samp, Grain
from .molecular import (
    molecular_pursuit
)
from .atomic import (
    decaycos_atomic_pursuit_precond,
    decaycos_ts_pursuit_raw
)

"""
The main functions here are the matching pursuits for atomic decompositions of
a particular spectrogram frame into decaying sinusoid atoms, and the molecular
decomposition of a spectrogram in terms of molecules.

The functions in this operate on structs and may be cached.
"""


class AutoCorrAtomicApproxSeries(SequenceFeature):
    """
    the storage of correlation bases is transposed with respect to librosa,
    which is to say, per default we iterate over time.

    We keep everything in a struct. This is the struct.
    """
    pass


class AutoCorrAtomicApprox(SampleStruct):
    """
    A single match in atoms
    """
    def __init__(self, coef, *args, **kwargs):
        self.coef = coef
        super().__init__(*args, **kwargs)


class GrainCodebook(SampleStruct):
    """
    A list of samples and locations
    """
    def __init__(self, entries=[], *args, **kwargs):
        self.entries = list(entries)
        super().__init__(*args, **kwargs)

    def grain(self, i):
        return self.entries[i]

    def sample(self, i):
        return self.entries[i].sample

    def t(self, i):
        return self.entries[i].t

    def acorr(self, i):
        return autocorr_t(self.sample(i), self.t(i)).feature

    def acorr_coef(self, i):
        return autocorr_pursuit_t(self.sample(i), self.t(i)).coef

    def grains(self):
        return self.entries

    def samples(self):
        return [
            self.sample(i) for i, _ in enumerate(self.entries)
        ]

    def ts(self):
        return [
            self.t(i) for i, _ in enumerate(self.entries)
        ]

    def acorrs(self):
        return [
            self.acorr(i) for i, _ in enumerate(self.entries)
        ]

    def acorr_coefs(self):
        return [
            self.acorr_coef(i) for i, _ in enumerate(self.entries)
        ]

    def append(self, grain):
        self.entries.append(grain)

    def __len__(self):
        return len(self.entries)


class AutoCorrMolecularApprox(GrainCodebook):
    """
    A single match in molecules
    """
    def __init__(
            self,
            entries=[],
            gains=None,
            rates=None,
            *args, **kwargs):
        super().__init__(entries, *args, **kwargs)
        if gains is None:
            gains = np.ones(len(entries))
        self._gains = gains
        if rates is None:
            rates = np.ones(len(entries))
        self._rates = rates

    def gain(self, i):
        return self._gains[i]

    def rate(self, i):
        return self._rates[i]

    def gains(self):
        return self._gains

    def rates(self):
        return self._rates


@memory.cache
def decaycos_ts_pursuit(
        sample,
        basis_size=5,
        hop_length=1024,
        frame_length=4096,
        delay_step_size=8,
        n_delays=128,
        **kwargs):

    corr = autocorr_featurize(
        sample,
        hop_length=hop_length,
        frame_length=frame_length,
        delay_step_size=delay_step_size,
        n_delays=n_delays,
    )
    # correlation is packed timewise;
    # (time_step, period)
    # unlike librosa spectrograms, which are
    # (period, time_step)
    # but there is much less mess if we have time as the 1st axis
    # in the approximand

    approx_array, loss = decaycos_ts_pursuit_raw(corr.features, basis_size)

    return AutoCorrAtomicApproxSeries(
        features=approx_array,
        loss=loss,
        frame_length=frame_length,
        delay_step_size=delay_step_size,
        n_delays=n_delays,
        hop_length=hop_length,
        start_samp=corr.start_samp,
        sr=corr.sr
    )


@memory.cache
def autocorr_pursuit_samp(
            sample,
            s,
            frame_length=4096,
            delay_step_size=8,
            n_delays=128,
            basis_size=5,
            **kwargs
        ):
    target = autocorr_samp(
        sample,
        s,
        frame_length=frame_length,
        delay_step_size=delay_step_size,
        n_delays=n_delays,
    )
    coef, loss, scale = decaycos_atomic_pursuit_precond(
        target.feature,
        basis_size=basis_size,
        # x0=f,
        **kwargs)
    return AutoCorrAtomicApprox(
        coef,
        loss=loss,
        scale=scale,
        frame_length=frame_length,
        n_delays=n_delays,
        sr=sample.sr,
        delay_step_size=delay_step_size,
    )


def autocorr_pursuit_t(
            sample,
            t,
            delay_step_size=8,
            **kwargs
        ):
    return autocorr_pursuit_samp(
        sample,
        sample.quantize_time_to_samp(
            t,
            quantum=delay_step_size,
            **kwargs),
        delay_step_size=delay_step_size,
        **kwargs)


def gen_sample_codebook(
        sample,
        code_size=5,
        a=0.0,
        c=1.0,
        min_power=1e-8,
        verbose=0):
    ts = prr(code_size, high=sample.duration(), a=a, c=c)
    if verbose >= 11:
        print('codetimes', ts)
    grains = []
    for t in ts:
        grain = Grain(sample, t)
        if grain.avg_power() > min_power:
            grains.append(grain)
    return GrainCodebook(
        grains
    )


def cat_sample_codebooks(
        codebooks):
    all_entries = []
    for c in codebooks:
        all_entries.extend(c.entries)
    return codebooks(all_entries)


def gen_multisample_codebook(
        samples,
        code_size,
        a=0.0,
        c=1.0):
    return cat_sample_codebooks(
        [
            gen_sample_codebook(s, a=a, c=c, code_size=code_size)
            for s in samples
        ]
    )


def match_from_sample(
        target_sample,
        target_t,
        source_sample,
        code_size=3,
        codebook_a=0.173,
        codebook_c=1.795,
        pitched=True,
        verbose=0,
        *args, **kwargs):
    if verbose >= 24:
        print('codebook_ac', codebook_a, codebook_c)
    codebook = gen_sample_codebook(
        source_sample,
        code_size,
        a=codebook_a,
        c=codebook_c,
        verbose=verbose)
    return match_from_codebook(
        target_sample,
        target_t,
        codebook,
        pitched=pitched,
        verbose=verbose,
        *args, **kwargs
    )


def match_from_codebook(
        target_sample,
        target_t,
        codebook,
        basis_size=3,
        verbose=0,
        pitched=True,
        **mp_args):
    target_frame = autocorr_t(
        target_sample, target_t).feature
    code_coefs = codebook.acorr_coefs()
    if np.allclose(code_coefs[0], code_coefs[1]) and verbose >= 4:
        print("COEFFICIENT SUSPICION at", target_t)
        print(code_coefs)

    gain_rate, molecule_idx, loss, scale = molecular_pursuit(
        target=target_frame,
        code_coefs=code_coefs,
        basis_size=basis_size,
        pitched=pitched,
        verbose=verbose,
        **mp_args)
    r = AutoCorrMolecularApprox(
        [codebook.grain(i) for i in molecule_idx],
        gains=gain_rate[0],
        rates=gain_rate[1],
        loss=loss,
        scale=scale,
        target_t=target_t,  # Blech. Needed for rendering.
    )
    if verbose >= 18:
        print("match_from_codebook", r)
    return r
