"""A wire-compatible Mojo port of Blosc2's blocked shuffle plus LZ4 path."""

from . import core as _core
from .core import (
    EXTENDED_HEADER_LENGTH,
    MAX_BLOCKSIZE,
    MAX_BUFFERSIZE,
    MAX_OVERHEAD,
    MAX_TYPESIZE,
    MIN_HEADER_LENGTH,
    Codec,
    Filter,
    SplitMode,
    Tuner,
    byte_shuffle,
    byte_unshuffle,
    clib_info,
    compress,
    compress2,
    compressor_list,
    decompress,
    decompress2,
    free_resources,
    get_blocksize,
    get_cbuffer_sizes,
    get_clib,
    pack,
    pack_array,
    set_blocksize,
    set_nthreads,
    unpack,
    unpack_array,
)


def __getattr__(name):
    if name == "nthreads":
        return _core.nthreads
    raise AttributeError(name)

__all__ = [
    "Codec",
    "Filter",
    "SplitMode",
    "Tuner",
    "byte_shuffle",
    "byte_unshuffle",
    "clib_info",
    "compress",
    "compress2",
    "compressor_list",
    "decompress",
    "decompress2",
    "free_resources",
    "get_blocksize",
    "get_cbuffer_sizes",
    "get_clib",
    "nthreads",
    "pack",
    "pack_array",
    "set_blocksize",
    "set_nthreads",
    "unpack",
    "unpack_array",
]
__version__ = "0.1.0"
