import numpy as np
from numpy.random import normal as rnormal, seed as rseed, randint
from itertools import chain, islice

from .autocorr_pursuit.main import match_from_sample
from . import sample
from . import pattern_iter, timed_stream
from .random import gamma_v
from joblib import Parallel, delayed
from math import sqrt
import traceback


def chunker(iterable, n):
    """
    max size chunks from iterable
    https://stackoverflow.com/a/8998040
    """
    while True:
        try:
            nxt = next(iterable)
        except StopIteration:
            break
        yield chain(
            (nxt,),
            islice(iterable, n-1))


def cross_render_sample_jitter(
        target_sample,
        source_sample,
        grain_duration=0.01,
        grain_dur_jitter=0.0,
        grain_interval_jitter=0.0,
        grain_jitter=0.0,  # proportional offset grain centers from each other
        density=4.0,
        verbose=0,
        code_size=4,
        hop_length=1024,
        frame_length=4096,
        delay_step_size=8,
        n_delays=128,
        source_anchor='left',  # or 'center', not actually implemented
        dest_anchor='center',  # or 'left'
        window='hann',
        n_workers=1,  # parallelism
        chunk_size=None,  # paralleism sync
        seamless=False,
        seed=None,
        random_flip=True,
        **kwargs):
    """
    random everything; but all grains from a single fit share the same
    target location.
    When anchor is 'center', grain locations are centres.
    """
    rseed(seed)
    if chunk_size is None:
        chunk_size = n_workers * 4
    output_sample = sample.zero_sample(
        duration=target_sample.duration(),
        sr=target_sample.sr,
        stem=source_sample.stem,
        parent_path=target_sample.parent_path)
    output_sample.append_stem('_x_')
    output_sample.append_stem(target_sample.stem)
    kwarg_fn = timed_stream.DictStream(
        kwargs
    )
    inter_onset_period = grain_duration / density
    dest_high = target_sample.duration()
    approx_last_step = dest_high / inter_onset_period
    if verbose >= 10:
        print("gp", inter_onset_period, density, grain_duration, code_size)
        print('dest_high', dest_high)
    if verbose >= 2:
        print('approx_last_step', approx_last_step)
    param_list = [
        (
            step,
            target_t,
            kwarg_fn(target_t)
        ) for step, target_t in
        enumerate(
            pattern_iter.gamma_renewal_proc_scale(
                interval=inter_onset_period,
                var=inter_onset_period * grain_interval_jitter**2,
                high=dest_high,
                verbose=verbose
            )
        )
    ]
    # I chunk this for intermediate results:
    # https://stackoverflow.com/a/43085080
    with Parallel(
            n_jobs=n_workers,
            verbose=verbose*3) as pool:
        step = 0
        target_t_prev = 0.0
        try:
            for chunk in chunker(iter(param_list), chunk_size):
                for local_step, match in enumerate(pool([
                        delayed(match_from_sample)(
                            target_sample=target_sample,
                            source_sample=source_sample,
                            target_t=target_t,
                            verbose=verbose,
                            hop_length=hop_length,
                            frame_length=frame_length,
                            delay_step_size=delay_step_size,
                            n_delays=n_delays,
                            anchor=source_anchor,
                            window=window,
                            code_size=code_size,
                            **kw
                        ) for step, target_t, kw
                        in chunk
                        ])):

                    grains = match.grains()
                    ac_gains = match.gains()
                    rates = match.rates()
                    # if verbose >= 2:
                    #     elapsed = time() - start_time()
                    #     print(
                    #         "step", step+1,
                    #         "/", approx_last_step, ",",
                    #         elapsed, "/",
                    #         elapsed * (step+1)/approx_last_step)
                    target_t = match.target_t
                    target_t_inc = target_t - target_t_prev
                    target_t_prev = target_t
                    if verbose >= 3:
                        print(
                            "step", step, local_step,
                            't {:.5f}+{:.5f}'.format(target_t, target_t_inc),
                            'loss {:.5f}*{:.5f}'.format(
                                match.loss, match.scale)
                            )
                    if verbose >= 16:
                        print("match ", match)
                    for grain, ac_gain, rate in zip(
                            grains, ac_gains, rates):
                        if ac_gain == 0.0:
                            continue
                        gain = sqrt(abs(ac_gain*rate))
                        if random_flip:
                            gain = gain * (2*randint(2) - 1)
                        dest_t = match.target_t + np.asscalar(
                            rnormal(loc=0.0, scale=grain_jitter, size=1),
                        )
                        if seamless:
                            base_duration = target_t_inc
                        else:
                            base_duration = inter_onset_period
                        this_grain_duration = min(
                            gamma_v(
                                base_duration * density,
                                grain_dur_jitter * base_duration * density
                            ),
                            5.0
                        )
                        if verbose >= 6:
                            print(
                                "grain",
                                'dest {:.5f}'.format(dest_t),
                                'duration {:.5f}*{:.5f}'.format(
                                    this_grain_duration,
                                    this_grain_duration/target_t_inc))
                        if verbose >= 19:
                            print("is", grain, gain, rate)
                        sample.overdub_t(
                            dest_sample=output_sample,
                            source_sample=grain.sample,
                            dest_t=dest_t,
                            source_t=grain.t,
                            duration=this_grain_duration,
                            rate=rate,
                            mul=gain,
                            window=window,
                        )
                    step += 1
        except StopIteration as e:
            print('some iterator broke {}'.format(e))
            traceback.print_tb(e.__traceback__)
        except KeyboardInterrupt:
            print('stopping early')

    return output_sample


def cross_render_sample_adaptive(
        target_sample,
        source_sample,
        density=2.0,
        verbose=0,
        code_size=4,
        hop_length=1024,
        frame_length=4096,
        delay_step_size=8,
        grain_jitter=0.0,  # proportional offset grain centers from each other
        n_delays=128,
        source_anchor='center',  # or 'left', maybe not actually implemented?
        dest_anchor='center',  # or 'left'
        analysis_window='cosine',
        synthesis_window='hann',
        seed=None,
        random_flip=True,
        match_rtol=0.01,  # search early stopping parameter; not implemented
        progress=False,
        **kwargs):
    """
    Fit one new match at a time.
    """
    rseed(seed)
    sr = target_sample.sr
    output_sample = sample.zero_sample(
        duration=target_sample.duration(),
        sr=sr,
        stem=source_sample.stem,
        parent_path=target_sample.parent_path)
    output_sample.append_stem('_x_')
    output_sample.append_stem(target_sample.stem)
    kwarg_fn = timed_stream.DictStream(
        kwargs
    )
    grain_duration = frame_length / sr
    inter_onset_period = grain_duration / density
    dest_high = target_sample.duration()
    approx_last_step = dest_high / inter_onset_period
    if verbose >= 10:
        print("gp", inter_onset_period, density, grain_duration, code_size)
        print('dest_high', dest_high)
    if verbose >= 2:
        print('approx_last_step', approx_last_step)

    step = 0
    target_t_prev = 0.0
    target_t = 0.0
    target_i_prev = 0
    target_i = 0
    early_stop = False

    while target_i < output_sample.end:
        try:
            target_i += hop_length
            target_t = target_i / sr
            kw = kwarg_fn(target_t)
            match = match_from_sample(
                target_sample=target_sample,
                source_sample=source_sample,
                target_t=target_t,
                hop_length=hop_length,
                frame_length=frame_length,
                delay_step_size=delay_step_size,
                n_delays=n_delays,
                anchor=source_anchor,
                window=analysis_window,
                code_size=code_size,
                match_rtol=match_rtol,
                verbose=verbose,
                **kw
            )
            grains = match.grains()
            ac_gains = match.gains()
            rates = match.rates()
            # if verbose >= 2:
            #     elapsed = time() - start_time()
            #     print(
            #         "step", step+1,
            #         "/", approx_last_step, ",",
            #         elapsed, "/",
            #         elapsed * (step+1)/approx_last_step)
            target_t_inc = target_t - target_t_prev
            target_t_prev = target_t
            target_i_prev = target_i
            if verbose >= 3:
                print(
                    "step", step,
                    't {:.5f}+{:.5f}'.format(
                        target_t,
                        target_t_inc),
                    'loss {:.5f}*{:.5f}'.format(
                        match.loss,
                        match.scale)
                    )
            if verbose >= 16:
                print("match ", match)
            for g_i, grain, ac_gain, rate in zip(
                    range(len(grains)),
                    grains, ac_gains, rates):
                if ac_gain == 0.0:
                    continue
                gain = sqrt(abs(ac_gain*rate))
                if random_flip:
                    gain = gain * (2*randint(2) - 1)
                dest_t = target_t + np.asscalar(
                    rnormal(
                        loc=0.0,
                        scale=grain_duration * grain_jitter,
                        size=1)
                )
                this_grain_duration = target_t_inc * density
                if verbose >= 8:
                    print(
                        "subgrain",  g_i,
                        'dest {:.5f}'.format(dest_t),
                        'gain {:.5f}'.format(gain),
                        'rate {:.5f}'.format(rate),
                        'duration {:.5f}*{:.5f}'.format(
                            this_grain_duration,
                            this_grain_duration/target_t_inc))
                if verbose >= 19:
                    print("is", grain, gain, rate)
                landscape = np.copy(output_sample.get_audio_data_t(
                    dest_t, duration=this_grain_duration,
                    anchor="center"))
                sample.overdub_t(
                    dest_sample=output_sample,
                    source_sample=grain.sample,
                    dest_t=dest_t,
                    source_t=grain.t,
                    duration=this_grain_duration,
                    rate=rate,
                    mul=gain,
                    window=synthesis_window,
                    source_anchor=source_anchor,
                    dest_anchor=dest_anchor,
                    verbose=verbose,
                )
                if verbose >= 6:
                    y_so_far = output_sample.get_audio_data_t(
                        0,
                        target_t+this_grain_duration
                    )
                    print(
                        "glitch", np.percentile(
                            np.abs(
                                np.gradient(
                                    y_so_far
                                )
                            ),
                            [50, 95, 99, 99.5]
                        )
                    )
            step += 1
            if progress:
                print(step, '/', approx_last_step)
        except StopIteration as e:
            print('some iterator broke {}'.format(e))
            traceback.print_tb(e.__traceback__)
        except KeyboardInterrupt:
            print('stopping early at {} -- {}'.format(step, target_t))
            early_stop = True
            break

    return output_sample
