"""Python API matching the covered low-level Python-Blosc2 subset."""

from __future__ import annotations

import ctypes
import operator
import pickle
import struct
from enum import IntEnum

from ._lib import (
    LibraryError,
    destination_buffer,
    lib,
    source_buffer,
    workspaces,
    writable_bytes,
)

MIN_HEADER_LENGTH = 16
EXTENDED_HEADER_LENGTH = 32
MAX_OVERHEAD = 32
MAX_BUFFERSIZE = 2_147_483_615
MAX_TYPESIZE = 255
MAX_BLOCKSIZE = 536_866_816


class Codec(IntEnum):
    BLOSCLZ = 0
    LZ4 = 1
    LZ4HC = 2
    ZLIB = 4
    ZSTD = 5
    NDLZ = 32
    ZFP_ACC = 33
    ZFP_PREC = 34
    ZFP_RATE = 35


class Filter(IntEnum):
    NOFILTER = 0
    SHUFFLE = 1
    BITSHUFFLE = 2
    DELTA = 3
    TRUNC_PREC = 4


class SplitMode(IntEnum):
    ALWAYS_SPLIT = 1
    NEVER_SPLIT = 2
    AUTO_SPLIT = 3
    FORWARD_COMPAT_SPLIT = 4


class Tuner(IntEnum):
    STUNE = 0
    BTUNE = 32


_blocksize = 0
nthreads = 1


def _exact_int(value, name: str) -> int:
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc


def _as_enum(value, enum_type, name):
    if not isinstance(value, enum_type):
        value = _exact_int(value, name)
    try:
        return enum_type(value)
    except ValueError as exc:
        raise ValueError(f"unsupported {name}: {value!r}") from exc


def _validate_typesize(typesize: int) -> int:
    typesize = _exact_int(typesize, "typesize")
    if not 1 <= typesize <= MAX_TYPESIZE:
        raise ValueError(f"typesize must be between 1 and {MAX_TYPESIZE}")
    return typesize


def _validate_clevel(clevel: int) -> int:
    clevel = _exact_int(clevel, "clevel")
    if not 0 <= clevel <= 9:
        raise ValueError("clevel must be between 0 and 9")
    return clevel


def _auto_blocksize(size: int, typesize: int, requested: int = 0) -> int:
    blocksize = _exact_int(requested or _blocksize or 256 * 1024, "blocksize")
    if not 1 <= blocksize <= MAX_BLOCKSIZE:
        raise ValueError(f"blocksize must be between 0 and {MAX_BLOCKSIZE}")
    if size:
        blocksize = min(blocksize, size)
    return max(blocksize, 1)


def _source_itemsize(source, typesize):
    if typesize is None:
        return _exact_int(getattr(source, "itemsize", 1), "typesize")
    return typesize


def _compress(
    source,
    *,
    typesize: int,
    clevel: int,
    filter_code: Filter,
    codec: Codec,
    blocksize: int = 0,
    threads: int = 1,
) -> bytes:
    if codec is not Codec.LZ4:
        raise ValueError("the Mojo encoder currently supports codec=Codec.LZ4 only")
    source_address, size, source_keepalive = source_buffer(source)
    if size > MAX_BUFFERSIZE:
        raise ValueError("source is too large for a Blosc2 chunk")
    native_blocksize = _auto_blocksize(size, typesize, blocksize)
    destination, destination_address = writable_bytes(size + MAX_OVERHEAD)
    scratch, scratch_export, table = workspaces(native_blocksize)
    result = lib().mbl_compress(
        source_address,
        size,
        destination_address,
        size + MAX_OVERHEAD,
        scratch_export.address,
        ctypes.addressof(table),
        typesize,
        native_blocksize,
        int(filter_code),
        clevel,
        threads,
    )
    _ = source_keepalive, scratch
    if result < 0:
        raise LibraryError(f"compression failed with error {result}")
    return destination[:result]


def compress(
    src: object,
    typesize: int = 8,
    clevel: int = 1,
    filter: Filter = Filter.SHUFFLE,
    codec: Codec = Codec.ZSTD,
    _ignore_multiple_size: bool = False,
) -> str | bytes:
    typesize = _validate_typesize(_source_itemsize(src, typesize))
    clevel = _validate_clevel(clevel)
    filter_code = _as_enum(filter, Filter, "filter")
    if filter_code not in (Filter.NOFILTER, Filter.SHUFFLE):
        raise ValueError("the Mojo encoder supports NOFILTER and SHUFFLE only")
    codec_code = _as_enum(codec, Codec, "codec")
    _, size, keepalive = source_buffer(src)
    _ = keepalive
    if size % typesize and not _ignore_multiple_size:
        raise ValueError("source length must be a multiple of typesize")
    return _compress(
        src,
        typesize=typesize,
        clevel=clevel,
        filter_code=filter_code,
        codec=codec_code,
        threads=nthreads,
    )


def _parse_header(src) -> tuple[object, dict[str, int]]:
    source_address, size, keepalive = source_buffer(src)
    if size < MIN_HEADER_LENGTH:
        raise ValueError("source is smaller than the minimum Blosc2 header")
    header = ctypes.string_at(source_address, min(size, EXTENDED_HEADER_LENGTH))
    version, versionlz, flags, typesize, nbytes, blocksize, cbytes = (
        struct.unpack_from("<BBBBiii", header)
    )
    extended = flags & 5 == 5
    header_size = EXTENDED_HEADER_LENGTH if extended else MIN_HEADER_LENGTH
    if size < header_size or cbytes < header_size or cbytes > size:
        raise RuntimeError("corrupt Blosc2 chunk size")
    if (
        typesize == 0
        or nbytes < 0
        or nbytes > MAX_BUFFERSIZE
        or blocksize <= 0
        or blocksize > MAX_BLOCKSIZE
    ):
        raise RuntimeError("corrupt Blosc2 chunk header")
    special = (header[31] >> 4) & 7 if extended else 0
    if special not in (0, 1):
        raise ValueError("this decoder supports ordinary and special-zero chunks")
    if not flags & 2 and not special:
        compformat = (flags >> 5) & 7
        if compformat != 1:
            raise ValueError("this decoder supports Blosc2 LZ4/LZ4HC chunks only")
        if extended:
            if any(header[16:21]) or header[21] not in (0, 1):
                raise ValueError(
                    "this decoder supports NOFILTER and final-stage SHUFFLE only"
                )
    return keepalive, {
        "address": source_address,
        "size": size,
        "version": version,
        "versionlz": versionlz,
        "flags": flags,
        "typesize": typesize,
        "nbytes": nbytes,
        "blocksize": blocksize,
        "cbytes": cbytes,
        "header_size": header_size,
    }


def _decompress_to(
    src, destination_address: int, capacity: int, threads: int
) -> int:
    source_keepalive, info = _parse_header(src)
    if capacity < info["nbytes"]:
        raise RuntimeError("destination buffer is too small")
    _, scratch_export, _ = workspaces(max(info["blocksize"], 1))
    result = lib().mbl_decompress(
        info["address"],
        info["size"],
        destination_address,
        capacity,
        scratch_export.address,
        threads,
    )
    _ = source_keepalive
    if result < 0:
        raise RuntimeError(f"could not decompress the data (error {result})")
    if result != info["nbytes"]:
        raise RuntimeError("decompressed size does not match the chunk header")
    return result


def _decompress(
    src: object,
    dst: object | bytearray,
    as_bytearray: bool,
    threads: int,
) -> str | bytes | bytearray | None:
    _, info = _parse_header(src)
    nbytes = info["nbytes"]
    if dst is not None:
        export = destination_buffer(dst)
        if export.size == 0:
            raise ValueError("the dst length must be greater than 0")
        _decompress_to(src, export.address, export.size, threads)
        return None
    if as_bytearray:
        result = bytearray(nbytes)
        if nbytes:
            export = destination_buffer(result)
            _decompress_to(src, export.address, export.size, threads)
        else:
            temporary, address = writable_bytes(1)
            _decompress_to(src, address, 1, threads)
            _ = temporary
        return result
    result, address = writable_bytes(nbytes)
    _decompress_to(src, address, max(nbytes, 1), threads)
    return result if nbytes else b""


def decompress(
    src: object, dst: object | bytearray = None, as_bytearray: bool = False
) -> str | bytes | bytearray | None:
    return _decompress(src, dst, as_bytearray, nthreads)


def compress2(src: object, **kwargs: dict) -> str | bytes:
    supported = {
        "codec",
        "codec_meta",
        "clevel",
        "use_dict",
        "typesize",
        "nthreads",
        "blocksize",
        "splitmode",
        "filters",
        "filters_meta",
        "tuner",
    }
    unknown = set(kwargs) - supported
    if unknown:
        raise TypeError(f"unsupported compression parameter: {sorted(unknown)[0]}")
    codec = _as_enum(kwargs.get("codec", Codec.ZSTD), Codec, "codec")
    typesize = _validate_typesize(kwargs.get("typesize", 8))
    clevel = _validate_clevel(kwargs.get("clevel", 5))
    if kwargs.get("codec_meta", 0) != 0:
        raise ValueError("codec_meta is not supported")
    if kwargs.get("use_dict", False):
        raise ValueError("compression dictionaries are not supported")
    threads = _exact_int(kwargs.get("nthreads", nthreads), "nthreads")
    if threads < 1:
        raise ValueError("nthreads must be positive")
    splitmode = _as_enum(
        kwargs.get("splitmode", SplitMode.AUTO_SPLIT), SplitMode, "splitmode"
    )
    if splitmode is not SplitMode.AUTO_SPLIT:
        raise ValueError("the Mojo encoder supports AUTO_SPLIT only")
    tuner = _as_enum(kwargs.get("tuner", Tuner.STUNE), Tuner, "tuner")
    if tuner is not Tuner.STUNE:
        raise ValueError("the Mojo encoder supports the STUNE tuner only")
    filters = kwargs.get(
        "filters",
        [Filter.NOFILTER] * 5 + [Filter.SHUFFLE],
    )
    if len(filters) != 6:
        raise ValueError("filters must contain exactly six entries")
    filter_values = [_as_enum(value, Filter, "filter") for value in filters]
    if any(value is not Filter.NOFILTER for value in filter_values[:5]):
        raise ValueError("only final-stage SHUFFLE is supported")
    if filter_values[5] not in (Filter.NOFILTER, Filter.SHUFFLE):
        raise ValueError("only final-stage SHUFFLE is supported")
    filters_meta = kwargs.get("filters_meta", [0] * 6)
    if len(filters_meta) != 6 or any(
        _exact_int(value, "filters_meta entry") for value in filters_meta
    ):
        raise ValueError("non-zero filters_meta values are not supported")
    source_address, size, keepalive = source_buffer(src)
    _ = source_address, keepalive
    if size % typesize:
        raise ValueError("source length must be a multiple of typesize")
    return _compress(
        src,
        typesize=typesize,
        clevel=clevel,
        filter_code=filter_values[5],
        codec=codec,
        blocksize=_exact_int(kwargs.get("blocksize", 0), "blocksize"),
        threads=threads,
    )


def decompress2(src: object, dst: object | bytearray = None, **kwargs: dict):
    unknown = set(kwargs) - {"nthreads"}
    if unknown:
        raise TypeError(f"unsupported decompression parameter: {sorted(unknown)[0]}")
    threads = _exact_int(kwargs.get("nthreads", nthreads), "nthreads")
    if threads < 1:
        raise ValueError("nthreads must be positive")
    return _decompress(src, dst, False, threads)


def pack(
    obj: object,
    clevel: int = 9,
    filter: Filter = Filter.SHUFFLE,
    codec: Codec = Codec.BLOSCLZ,
) -> str | bytes:
    if not hasattr(obj, "itemsize"):
        raise AttributeError("The object must have an itemsize attribute.")
    if not hasattr(obj, "size"):
        raise AttributeError("The object must have an size attribute.")
    payload = pickle.dumps(obj, pickle.HIGHEST_PROTOCOL)
    return compress(
        payload,
        typesize=_exact_int(obj.itemsize, "itemsize"),
        clevel=clevel,
        filter=filter,
        codec=codec,
        _ignore_multiple_size=True,
    )


def unpack(packed_object: str | bytes, **kwargs: dict) -> object:
    return pickle.loads(decompress(packed_object), **kwargs)


def pack_array(
    arr,
    clevel: int = 9,
    filter: Filter = Filter.SHUFFLE,
    codec: Codec = Codec.BLOSCLZ,
) -> str | bytes:
    return pack(arr, clevel, filter, codec)


def unpack_array(packed_array: str | bytes, **kwargs: dict):
    return unpack(packed_array, **kwargs)


def get_cbuffer_sizes(src: object) -> tuple[int, int, int]:
    _, info = _parse_header(src)
    return info["nbytes"], info["cbytes"], info["blocksize"]


def get_clib(src: object) -> str:
    _, info = _parse_header(src)
    names = {0: "BloscLZ", 1: "LZ4", 3: "Zlib", 4: "Zstd"}
    return names.get((info["flags"] >> 5) & 7, "unknown")


def compressor_list(plugins: bool = False) -> list[str]:
    _ = plugins
    return ["lz4"]


def clib_info(codec) -> tuple[str, str]:
    if _as_enum(codec, Codec, "codec") is not Codec.LZ4:
        raise ValueError("only the LZ4 codec is implemented")
    return "LZ4", "Mojo native LZ4 block codec"


def get_blocksize() -> int:
    return _blocksize


def set_blocksize(blocksize: int = 0) -> None:
    global _blocksize
    blocksize = _exact_int(blocksize, "blocksize")
    if not 0 <= blocksize <= MAX_BLOCKSIZE:
        raise ValueError(f"blocksize must be between 0 and {MAX_BLOCKSIZE}")
    _blocksize = blocksize


def set_nthreads(value: int) -> int:
    global nthreads
    value = _exact_int(value, "nthreads")
    if value < 1:
        raise ValueError("nthreads must be positive")
    previous = nthreads
    nthreads = value
    return previous


def free_resources() -> None:
    return None


def byte_shuffle(src: object, typesize: int, nthreads: int = 1) -> bytes:
    typesize = _validate_typesize(typesize)
    nthreads = _exact_int(nthreads, "nthreads")
    if nthreads < 1:
        raise ValueError("nthreads must be positive")
    source_address, size, source_keepalive = source_buffer(src)
    result, result_address = writable_bytes(size)
    status = lib().mbl_shuffle(
        source_address, result_address, size, typesize, nthreads
    )
    _ = source_keepalive
    if status:
        raise LibraryError(f"shuffle failed with error {status}")
    return result if size else b""


def byte_unshuffle(src: object, typesize: int, nthreads: int = 1) -> bytes:
    typesize = _validate_typesize(typesize)
    nthreads = _exact_int(nthreads, "nthreads")
    if nthreads < 1:
        raise ValueError("nthreads must be positive")
    source_address, size, source_keepalive = source_buffer(src)
    result, result_address = writable_bytes(size)
    status = lib().mbl_unshuffle(
        source_address, result_address, size, typesize, nthreads
    )
    _ = source_keepalive
    if status:
        raise LibraryError(f"unshuffle failed with error {status}")
    return result if size else b""
