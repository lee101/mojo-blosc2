import inspect
import struct

import blosc2 as upstream
import numpy as np
import pytest

import mojo_blosc2 as mojo


def upstream_codec(value):
    return upstream.Codec(int(value))


def upstream_filter(value):
    return upstream.Filter(int(value))


def reference_shuffle(source, typesize):
    data = bytes(source)
    elements, remainder = divmod(len(data), typesize)
    shuffled = bytearray(len(data))
    for byte_index in range(typesize):
        for element in range(elements):
            shuffled[byte_index * elements + element] = data[
                element * typesize + byte_index
            ]
    if remainder:
        shuffled[-remainder:] = data[-remainder:]
    return bytes(shuffled)


def test_public_signatures_match_upstream():
    for name in ("compress", "decompress", "pack", "pack_array"):
        assert tuple(inspect.signature(getattr(mojo, name)).parameters) == tuple(
            inspect.signature(getattr(upstream, name)).parameters
        )


def test_enum_values_match_upstream():
    assert mojo.Codec.LZ4.value == upstream.Codec.LZ4.value
    assert mojo.Filter.SHUFFLE.value == upstream.Filter.SHUFFLE.value
    assert mojo.SplitMode.NEVER_SPLIT.value == upstream.SplitMode.NEVER_SPLIT.value


@pytest.mark.parametrize("typesize", [1, 2, 3, 4, 8, 16])
def test_byte_shuffle_matches_reference(typesize):
    source = bytes((i * 29 + 7) & 255 for i in range(1003))
    expected = reference_shuffle(source, typesize)
    assert mojo.byte_shuffle(source, typesize) == expected
    assert mojo.byte_unshuffle(expected, typesize) == source


@pytest.mark.parametrize("typesize", [4, 8])
def test_simd_shuffle_unaligned_source_and_scalar_tail(typesize):
    storage = bytearray((i * 17 + 11) & 255 for i in range(1012))
    source = memoryview(storage)[1:1004]
    expected = reference_shuffle(source, typesize)
    assert mojo.byte_shuffle(source, typesize) == expected
    assert mojo.byte_unshuffle(expected, typesize) == bytes(source)


@pytest.mark.parametrize("size", [32 * 1024 * 1024 - 4, 32 * 1024 * 1024])
def test_shuffle_parallel_threshold_matches_serial(size):
    source = np.arange(size // 4, dtype=np.uint32)
    expected = mojo.byte_shuffle(source, 4, nthreads=1)
    actual = mojo.byte_shuffle(source, 4, nthreads=4)
    assert actual == expected
    assert mojo.byte_unshuffle(actual, 4, nthreads=4) == source.tobytes()


@pytest.mark.parametrize("size", [0, 1, 15, 31, 32, 100, 4096, 300_000])
@pytest.mark.parametrize("filter_code", [mojo.Filter.NOFILTER, mojo.Filter.SHUFFLE])
def test_roundtrip_across_sizes(size, filter_code):
    source = (b"blocked Mojo Blosc2 data " * (size // 25 + 1))[:size]
    encoded = mojo.compress(
        source,
        typesize=1,
        filter=filter_code,
        codec=mojo.Codec.LZ4,
    )
    assert mojo.decompress(encoded) == source


@pytest.mark.parametrize("dtype", [np.int16, np.int32, np.int64, np.float32, np.float64])
def test_mojo_chunks_decode_upstream(dtype):
    source = np.arange(250_003, dtype=dtype)
    encoded = mojo.compress(
        source,
        typesize=source.itemsize,
        codec=mojo.Codec.LZ4,
    )
    assert upstream.decompress(encoded) == source.tobytes()
    assert mojo.get_clib(encoded) in ("LZ4", "BloscLZ")


@pytest.mark.parametrize("size", [32, 1000, 100_000, 1_000_000])
@pytest.mark.parametrize(
    "filter_code", [upstream.Filter.NOFILTER, upstream.Filter.SHUFFLE]
)
def test_upstream_chunks_decode_mojo(size, filter_code):
    source = np.arange(size // 4, dtype=np.int32)
    encoded = upstream.compress(
        source,
        typesize=4,
        filter=filter_code,
        codec=upstream.Codec.LZ4,
    )
    assert mojo.decompress(encoded) == source.tobytes()


def test_split_stream_runs_from_upstream_decode_mojo():
    source = np.arange(500_000, dtype=np.int32)
    encoded = upstream.compress(
        source,
        typesize=4,
        filter=upstream.Filter.SHUFFLE,
        codec=upstream.Codec.LZ4,
    )
    assert not encoded[2] & 16
    assert mojo.decompress(encoded) == source.tobytes()


def test_special_zero_chunk_from_upstream_decode_mojo():
    source = bytes(1_000_000)
    encoded = upstream.compress(
        source, typesize=8, codec=upstream.Codec.LZ4
    )
    assert len(encoded) == mojo.EXTENDED_HEADER_LENGTH
    assert mojo.decompress(encoded) == source


def test_mojo_special_zero_chunk_decodes_upstream():
    source = np.zeros(250_000, dtype=np.int32)
    encoded = mojo.compress(
        source, typesize=4, codec=mojo.Codec.LZ4
    )
    assert len(encoded) == mojo.EXTENDED_HEADER_LENGTH
    assert upstream.decompress(encoded) == source.tobytes()


def test_incompressible_input_uses_standard_memcpy_chunk():
    source = np.random.default_rng(4).bytes(200_000)
    encoded = mojo.compress(
        source,
        typesize=1,
        filter=mojo.Filter.NOFILTER,
        codec=mojo.Codec.LZ4,
    )
    assert encoded[2] & 2
    assert len(encoded) == len(source) + mojo.MAX_OVERHEAD
    assert upstream.decompress(encoded) == source


def test_compress2_cross_compatibility_with_explicit_pipeline():
    source = np.arange(300_000, dtype=np.int64)
    encoded = mojo.compress2(
        source,
        codec=mojo.Codec.LZ4,
        typesize=8,
        clevel=5,
        blocksize=128 * 1024,
        nthreads=1,
        splitmode=mojo.SplitMode.AUTO_SPLIT,
        filters=[mojo.Filter.NOFILTER] * 5 + [mojo.Filter.SHUFFLE],
        filters_meta=[0] * 6,
    )
    assert upstream.decompress2(encoded) == source.tobytes()
    assert mojo.get_cbuffer_sizes(encoded)[0] == source.nbytes


def test_decompress2_accepts_upstream_never_split_chunk():
    source = np.linspace(-10, 10, 200_000)
    encoded = upstream.compress2(
        source,
        codec=upstream.Codec.LZ4,
        typesize=8,
        blocksize=256 * 1024,
        nthreads=1,
        splitmode=upstream.SplitMode.NEVER_SPLIT,
        filters=[upstream.Filter.NOFILTER] * 5 + [upstream.Filter.SHUFFLE],
        filters_meta=[0] * 6,
    )
    assert mojo.decompress2(encoded, nthreads=1) == source.tobytes()


def test_decompress_destination_and_bytearray_behavior():
    source = np.arange(50_000, dtype=np.uint32)
    encoded = mojo.compress(
        source, typesize=4, codec=mojo.Codec.LZ4
    )
    destination = np.empty_like(source)
    assert mojo.decompress(encoded, dst=destination) is None
    assert np.array_equal(destination, source)
    result = mojo.decompress(encoded, as_bytearray=True)
    assert isinstance(result, bytearray)
    assert result == source.tobytes()


def test_pack_array_cross_compatibility():
    source = np.arange(10_000, dtype=np.int32).reshape(100, 100)
    mojo_packed = mojo.pack_array(source, codec=mojo.Codec.LZ4)
    assert np.array_equal(upstream.unpack_array(mojo_packed), source)
    upstream_packed = upstream.pack_array(source, codec=upstream.Codec.LZ4)
    assert np.array_equal(mojo.unpack_array(upstream_packed), source)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"typesize": 0, "codec": mojo.Codec.LZ4}, "typesize"),
        ({"typesize": 4, "codec": mojo.Codec.LZ4}, "multiple"),
        ({"typesize": 1, "clevel": 10, "codec": mojo.Codec.LZ4}, "clevel"),
        ({"typesize": 1, "codec": mojo.Codec.ZSTD}, "LZ4"),
        (
            {
                "typesize": 1,
                "filter": mojo.Filter.BITSHUFFLE,
                "codec": mojo.Codec.LZ4,
            },
            "NOFILTER",
        ),
    ],
)
def test_compress_validation(kwargs, message):
    with pytest.raises(ValueError, match=message):
        mojo.compress(b"abc", **kwargs)


def test_default_codec_is_rejected_instead_of_silently_substituted():
    with pytest.raises(ValueError, match="LZ4"):
        mojo.compress(bytes(32))
    with pytest.raises(ValueError, match="LZ4"):
        mojo.compress2(bytes(32))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"splitmode": mojo.SplitMode.NEVER_SPLIT},
        {"tuner": mojo.Tuner.BTUNE},
        {
            "filters": [mojo.Filter.SHUFFLE]
            + [mojo.Filter.NOFILTER] * 5
        },
    ],
)
def test_compress2_rejects_parameters_it_cannot_honor(kwargs):
    with pytest.raises(ValueError):
        mojo.compress2(
            bytes(32),
            codec=mojo.Codec.LZ4,
            typesize=1,
            **kwargs,
        )


def test_noncontiguous_source_and_readonly_destination_rejected():
    source = np.arange(100, dtype=np.uint8)[::2]
    with pytest.raises(BufferError, match="contiguous"):
        mojo.compress(source, typesize=1, codec=mojo.Codec.LZ4)
    encoded = mojo.compress(
        bytes(range(100)), typesize=1, codec=mojo.Codec.LZ4
    )
    with pytest.raises(TypeError, match="read-only"):
        mojo.decompress(encoded, dst=bytes(100))


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (mojo.compress, (b"abcd",)),
        (mojo.byte_shuffle, (b"abcd",)),
        (mojo.set_blocksize, ()),
        (mojo.set_nthreads, ()),
    ],
)
def test_integer_parameters_are_not_silently_narrowed(function, args):
    keyword = {
        mojo.compress: {"typesize": 1.5, "codec": mojo.Codec.LZ4},
        mojo.byte_shuffle: {"typesize": 1.5},
        mojo.set_blocksize: {"blocksize": 1.5},
        mojo.set_nthreads: {"value": 1.5},
    }[function]
    with pytest.raises(TypeError, match="integer"):
        function(*args, **keyword)


def test_enum_parameters_are_not_silently_narrowed():
    with pytest.raises(TypeError, match="integer"):
        mojo.compress(b"abcd", typesize=1, codec=1.0)


def test_native_exports_reject_null_pointers():
    library = mojo_blosc2_library()
    assert library.mbl_compress(*([0] * 11)) < 0
    assert library.mbl_decompress(*([0] * 6)) < 0
    assert library.mbl_shuffle(0, 0, 1, 1, 1) < 0
    assert library.mbl_unshuffle(0, 0, 1, 1, 1) < 0


def mojo_blosc2_library():
    from mojo_blosc2._lib import lib

    return lib()


def test_small_destination_and_corrupt_headers_rejected():
    encoded = mojo.compress(
        bytes(range(100)), typesize=1, codec=mojo.Codec.LZ4
    )
    with pytest.raises(RuntimeError, match="too small"):
        mojo.decompress(encoded, dst=bytearray(10))
    with pytest.raises(ValueError, match="minimum"):
        mojo.decompress(b"short")
    corrupt = bytearray(encoded)
    struct.pack_into("<i", corrupt, 12, len(corrupt) + 1)
    with pytest.raises(RuntimeError, match="chunk size"):
        mojo.decompress(corrupt)
    oversized = bytearray(encoded)
    struct.pack_into("<i", oversized, 4, mojo.MAX_BUFFERSIZE + 1)
    with pytest.raises(RuntimeError, match="header"):
        mojo.decompress(oversized)


def test_unsupported_upstream_codec_is_reported():
    encoded = upstream.compress(
        np.arange(1000, dtype=np.int32),
        typesize=4,
        codec=upstream.Codec.ZSTD,
    )
    with pytest.raises(ValueError, match="LZ4"):
        mojo.decompress(encoded)


def test_global_blocksize_and_thread_compatibility_helpers():
    previous = mojo.get_blocksize()
    try:
        assert mojo.set_blocksize(64 * 1024) is None
        assert mojo.get_blocksize() == 64 * 1024
        encoded = mojo.compress(
            np.arange(100_000, dtype=np.int32),
            typesize=4,
            codec=mojo.Codec.LZ4,
        )
        assert mojo.get_cbuffer_sizes(encoded)[2] == 64 * 1024
    finally:
        mojo.set_blocksize(previous)
    assert mojo.set_nthreads(2) >= 1
    assert mojo.nthreads == 2
    assert mojo.set_nthreads(1) == 2


def test_advertised_utility_functions():
    source = mojo.compress(
        bytes(range(100)), typesize=1, codec=mojo.Codec.LZ4
    )
    assert mojo.get_cbuffer_sizes(source) == (100, len(source), 100)
    assert mojo.get_clib(source) == "LZ4"
    assert mojo.compressor_list() == ["lz4"]
    assert mojo.clib_info(mojo.Codec.LZ4)[0] == "LZ4"
    assert mojo.free_resources() is None


def test_pack_and_unpack_cross_compatibility():
    source = np.arange(101, dtype=np.int16)
    mojo_packed = mojo.pack(source, codec=mojo.Codec.LZ4)
    assert np.array_equal(upstream.unpack(mojo_packed), source)
    upstream_packed = upstream.pack(source, codec=upstream.Codec.LZ4)
    assert np.array_equal(mojo.unpack(upstream_packed), source)
