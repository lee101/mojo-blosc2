# mojo-blosc2

`mojo-blosc2` implements Blosc2's blocked byte-shuffle plus LZ4 compression
path in [Mojo](https://www.modular.com/mojo), with a Python API shaped like the
covered low-level [`blosc2`](https://www.blosc.org/python-blosc2/) functions.
It produces real Blosc2 chunks rather than a private look-alike format.

Chunks are interoperable in both directions with Python-Blosc2 4.9.1. Upstream
can decompress Mojo-produced chunks, and Mojo can decompress upstream LZ4
chunks using split or unsplit streams, byte shuffle or no filter, raw memcpy
fallback, byte-run markers, and the special-zero representation.

```python
import numpy as np
import mojo_blosc2 as blosc2

values = np.arange(250_000, dtype=np.int32)
compressed = blosc2.compress(
    values,
    typesize=values.itemsize,
    filter=blosc2.Filter.SHUFFLE,
    codec=blosc2.Codec.LZ4,
)
restored = np.frombuffer(blosc2.decompress(compressed), dtype=values.dtype)
assert np.array_equal(restored, values)
```

## Coverage

| Python-Blosc2 area | Covered subset |
| --- | --- |
| one-shot buffers | `compress`, `decompress`, destination buffers, `as_bytearray` |
| context-style buffers | `compress2`, `decompress2`, the supported parameters listed below |
| Python objects | `pack`, `unpack`, `pack_array`, `unpack_array` |
| utilities | `get_cbuffer_sizes`, `get_clib`, `compressor_list`, `clib_info`, block-size and thread compatibility setters |
| codec | LZ4 compression; LZ4 and LZ4HC wire-format decompression |
| filters | byte `SHUFFLE` and `NOFILTER` |
| chunk encodings | blocked split/unsplit streams, raw streams, memcpy chunks, byte runs, special zero |

The public function signatures retain upstream defaults. Upstream defaults
`compress` and `compress2` to ZSTD, which is outside this repository's scope,
so callers must explicitly pass `codec=Codec.LZ4`. Unsupported codecs and
filters raise an error; they are never silently replaced. The same applies to
the BLOSCLZ default on `pack` and `pack_array`. The encoder accepts STUNE,
AUTO_SPLIT, and either NOFILTER or final-stage SHUFFLE; other tuner, split, and
filter-pipeline settings are rejected instead of ignored.

Not covered are ZSTD, BLOSCLZ, Zlib, LZ4HC encoding, bitshuffle, delta and lossy
filters, dictionaries, parallel LZ4 parsing, user plugins, `SChunk`, compressed
frames, `NDArray`, lazy expressions, and buffers larger than one Blosc2 chunk.
The `nthreads` parameter parallelizes independent shuffle ranges for blocks at
least 32 MiB; smaller blocks stay serial because thread launch overhead is
larger than the saved work. Compression levels select the LZ4 search
acceleration; they do not provide an HC parser.

`byte_shuffle` and `byte_unshuffle` are additional direct helpers for the
filter kernel. They preserve any final bytes that do not form a complete
element, matching Blosc2's generic shuffle rule.

## Install and run

The repository pins its Mojo nightly and manages Python-Blosc2 from PyPI for
parity tests:

```bash
pixi install
pixi run build
pixi run test
pixi run bench
```

`pixi run build` writes `dist/libmojo-blosc2.so`. Pixi sets
`PYTHONPATH=python`, so the usage example runs from the checkout without a
wheel installation.

## Performance

Measured with `pixi run bench` on an Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
x86-64, Python 3.13.14, Mojo `1.0.0b3.dev2026072406`, and upstream blosc2
4.9.1. Both implementations use one thread, LZ4, 256 KiB blocks, compression
level 5, and the same filter selection. Each value is the best of seven timed
runs after one warmup. Decompression uses the exact same upstream-produced
chunk. Relative is upstream time divided by Mojo time, so values below 1.00
mean Mojo is slower and values above 1.00 mean Mojo is faster.

| case | mojo-blosc2 | upstream blosc2 | relative |
| --- | ---: | ---: | ---: |
| compress int32 arange, 8 MB | 2.33 ms | 2.49 ms | 1.07x |
| decompress int32 arange, 8 MB | 4.35 ms | 1.37 ms | 0.32x |
| compress smooth float64, 16 MB | 20.88 ms | 14.13 ms | 0.68x |
| decompress smooth float64, 16 MB | 9.94 ms | 4.42 ms | 0.44x |
| compress random bytes, 8 MiB | 2.99 ms | 2.26 ms | 0.76x |
| decompress memcpy chunk, 8 MiB | 1.06 ms | 1.16 ms | 1.10x |

For the thresholded direct helpers, a 64 MB int32 buffer shuffled in 51.18 ms
with one worker and 27.00 ms with four; unshuffle moved from 43.92 ms to
24.68 ms. Smaller inputs remain serial.

There is intentionally no GPU path. Shuffle, fill, copies, and hash-table
clearing move far more bytes than arithmetic operations, while LZ4 parsing is
branch-dependent and sequential within each stream. None reaches the roughly
two-flops-per-byte threshold where device transfer and launch costs would be
justified.

## How it works

`src/blosc2.mojo` is one compilation unit containing the SIMD byte transpose,
greedy LZ4 block encoder, bounds-checked LZ4 decoder, stream run handling, and
the Blosc2 chunk reader/writer. Compression divides the source into cache-sized
blocks, shuffles complete elements into byte planes, optionally splits those
planes into separate LZ4 streams, and writes the standard 32-byte extended
header plus block-start table. The encoder uses generation-tagged hash entries
across small split streams instead of clearing the full table for every stream.
Incompressible input becomes a standard memcpy chunk rather than expanding
beyond Blosc2's 32-byte maximum overhead.

Python owns every allocation. Contiguous source buffers, newly allocated
destination storage, one reusable block scratch buffer, and a reusable
65,536-entry LZ4 hash table cross the C ABI as integer addresses. Mojo rebuilds
them as `UnsafePointer[..., AnyOrigin[mut=True]]`, retains no pointer after the
call, and performs no heap allocation. NumPy arrays and other contiguous
buffer-protocol objects therefore cross without first becoming Python
`bytes`; compressed output is trimmed to its final chunk length on return.

The decoder reads little-endian chunk metadata, validates all sizes and block
offsets, expands each raw, run-length, or LZ4 stream into scratch memory, then
reverses byte shuffle into the caller's output buffer. Tests assert
bidirectional wire compatibility rather than only testing self-round-trips.

## License

MIT
