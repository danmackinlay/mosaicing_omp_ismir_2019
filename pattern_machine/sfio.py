"""
File IO using FFMPEG.

Slow because of invoking external util,
but not as slow as me adding 24 bit support to audioread.

For alternative approaches, see Tensorflow's FFMPEG loader,
and if your files are not too challenging the built-in librosa loader is fine,
but it's pysndfile support is better,
"""
import tempfile
import subprocess as sp
import os
import os.path
import librosa
from pathlib import Path
import logging
import base64
import time
import numpy as np
from librosa.util import normalize

_tmp_dir = None

# librosa does fine on these
librosa_file_extensions = {"wav", "wave"}
# Sox will do these well
sox_file_extensions = {"aif", "aiff", "wav", "wave"}
# For the rest, we fall back to ffmpeg.


def load(filename, sr=44100, nchan=1, offset=0.0, duration=None, **kwargs):
    """
    We never use librosa's importer with default settings
    because it will erroneously load 24 bit aiffs as 24 bit wavs and explode
    without raising an error.
    """
    path = Path(filename)
    if not path.exists():
        raise FileNotFoundError(filename)
    extn = path.suffix
    if len(extn):
        extn = str.lower(extn[1:])
    if extn in librosa_file_extensions:
        return librosa.core.load(
            str(path),
            sr=sr,
            mono=True if nchan == 1 else 2,
            offset=offset,
            duration=duration,
            **kwargs)
    else:
        return _load_ffmpeg_pipe(
            str(path),
            sr=sr,
            nchan=nchan,
            offset=offset,
            duration=duration,
            **kwargs), sr


def _info_ffmpeg(filepath):
    # ffprobe -v error -show_streams -select_streams a:0 -of json thumb_piano.mp3
    # ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 thumb_piano.mp3
    pass


def _load_ffmpeg_pipe(filepath, sr=44100, offset=0.0, duration=30.0, nchan=1):
    """Fastest loading method, but enforces numbr of channels and sample rate.
    TODO: find file info with ffprobe
    https://stackoverflow.com/a/22243834
    If our input file is 5.1 something will probably go nasty.
    """
    command = [
        'ffmpeg',
        '-ss', str(float(offset)),
        '-t', str(float(duration)),
        '-i', str(filepath),
        '-acodec', 'pcm_f32le',
        '-f', 'f32le',
        '-ar', str(sr),
        '-ac', str(nchan),
        "-"
    ]
    bufsize = int(4 * duration * sr * nchan)  # up to 4 bytes width
    proc = sp.run(
        command,
        stdout=sp.PIPE,
        bufsize=bufsize,
        stderr=sp.DEVNULL,
        check=True)
    return np.fromstring(proc.stdout, dtype="float32")


def _save_ffmpeg(
        filepath,
        data,
        norm=True,
        sr=44100,
        quality=2,
        format="s16",
        **kwarg):
    """
    the twin to _load_ffmpeg.
    note we don't get to specify endianness in the `format`,
    since we are not working with raw data.
    This is asymmetric with respect to _load_ffmpeg.
    Also 32 bit float is "flt".
    Run `ffmpeg -sample_fmts` to see the menu.
    """
    filepath = Path(filepath)
    ext = filepath.suffix
    if norm:
        data = normalize(data, axis=-1)
    if len(data.shape) == 1:
        nchan = 1
    else:
        nchan = data.shape[0]  # librosa convention channels first.
    if ext == '.mp3':
        # Ignore format - it's always "s16"``
        command = [
            'ffmpeg',
            '-y',
            '-f', 'f32le',
            '-acodec', 'pcm_f32le',
            '-ac', str(nchan),
            '-ar', str(sr),
            '-i', '-',
            '-acodec', 'libmp3lame',
            '-ar', str(sr),
            '-ac', str(nchan),
            '-aq', str(quality),
            str(filepath)
        ]
    else:
        command = [
            'ffmpeg',
            '-y',
            '-f', 'f32le',
            '-acodec', 'pcm_f32le',
            '-ac', str(nchan),
            '-ar', str(sr),
            '-i', '-',
            '-sample_fmt', format,
            '-ar', str(sr),
            '-ac', str(nchan),
            '-aq', str(quality),
            str(filepath)
        ]
    data = data.astype('float32', copy=False)
    bufsize = data.itemsize * data.size
    return sp.run(
        command,
        input=data.tobytes(),
        bufsize=bufsize,
        capture_output=True,
        check=True)


def norm_path(path):
    return Path(path).expanduser()


def save(
        filename,
        y,
        sr=44100,
        format="s16le",  # 24 bit signed per default
        norm=True,
        **kwargs):
    # librosa saves using scipy, which makes fat 32-bit float wavs
    # these are big and inconvenient.
    extn = Path(filename).suffix
    if len(extn):
        extn = str.lower(extn[1:])

    if extn in librosa_file_extensions and format == "f32le":
        librosa.output.write_wav(
            filename,
            y.astype('float32'),
            sr=sr,
            norm=norm
        )

    else:
        return _save_ffmpeg(
            filename,
            y,
            sr=sr,
            format=format,
            norm=norm,
            **kwargs)


def unique_output_for(output_dir, source_file, suffix=''):
    unique_part = (
        Path(safeish_hash(source_file))
    )
    last_part = unique_part / Path(
        Path(source_file).name
    ).with_suffix(suffix)
    full_path = norm_path(output_dir) / last_part
    (full_path.parent).mkdir(exist_ok=True, parents=True)
    return last_part, full_path


def get_tmp_dir():
    global _tmp_dir
    if _tmp_dir is None:
        _tmp_dir = tempfile.mkdtemp(prefix="easy_listener")
    return _tmp_dir


def safeish_hash(obj, n=8):
    """
    return a short path-safe string hash of an object based on its repr value
    """
    return base64.urlsafe_b64encode(
        (
            hash(repr(obj)) % (2**32)
        ).tobytes(4, byteorder='big', signed=False)
    ).decode("ascii")[:n]


def timestring(prefix=''):
    return '{}{}'.format(
        prefix,
        time.strftime("%Y%m%d-%H%M%S")
    )
