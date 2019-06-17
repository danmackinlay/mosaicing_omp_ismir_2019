"""
Joblib cache, handles probably-not-async-compatible caching to disk
"""
import base64
import hashlib
import os
from pathlib import Path
from tempfile import mkdtemp
import warnings

import dotenv
from joblib import Memory

dotenv.load_dotenv(dotenv.find_dotenv())

memory = None

cache_size = int(float(os.environ.get('PS_CACHE_SIZE', "0")))
cache_verbose = int(float(os.environ.get('PS_CACHE_VERBOSE', "0")))
cache_subcache = os.environ.get('PS_CACHE_SUBCACHE', "default")

cache_dir = os.environ.get('PS_CACHE_DIR', None)
if cache_dir is None:
    cache_dir = Path(mkdtemp())
else:
    cache_dir = (
        Path(os.environ.get('PS_CACHE_DIR', "")).expanduser() /
        cache_subcache
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

# put this back into the environment so the user can find autogen'd paths
os.environ['PS_CACHE_DIR'] = str(cache_dir)


def clear():
    import shutil
    for c in cache_dir.glob('*'):
        if cache_verbose > 0:
            print('purging {}'.format(c))
            if c.is_dir():
                shutil.rmtree(c)
            else:
                c.unlink()
    _init()


class DummyMemory:
    def cache(self, arg):
        return arg

    def clear(self):
        return


def _init():
    global memory
    if cache_size > 0:
        memory = Memory(
            cachedir=str(cache_dir),
            verbose=cache_verbose,
            mmap_mode='r',
            bytes_limit=cache_size
        )
        if cache_verbose > 0:
            warnings.warn(
                "caching in {}, {} bytes".format(
                    cache_dir,
                    cache_size,
                )
            )
    else:
        memory = DummyMemory()
        if cache_verbose > 0:
            warnings.warn("dummy caching ")


def strid(y):
    """
    filename-safe ID, with useless last char trimmed off
    """
    return base64.urlsafe_b64encode(hash_content(y)).decode()[:-1]


def hash_content(y):
    return hashlib.sha1(y).digest()


_init()