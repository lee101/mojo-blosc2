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
`nthreads` splits the shuffle into independent element ranges at 32 MiB and
above; the shuffle is a pure byte-plane gather/scatter, so each range runs on
the calling thread and smaller blocks stay serial because the split would cost
more than it saves.
Compression levels select the LZ4 search acceleration; they do not provide an
HC parser.

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
x86-64, Python 3.13.14, Mojo `1.2.0.dev2026092605` (AVX2, 32-byte vectors), and
upstream blosc2 4.9.1. Both implementations use one thread, LZ4, 256 KiB
blocks, compression level 5, and the same filter selection. Each value is the
best of seven timed runs after one warmup. Decompression uses the exact same
upstream-produced chunk. Relative is upstream time divided by Mojo time, so
values below 1.00 mean Mojo is slower and values above 1.00 mean Mojo is
faster. `before` is the same case measured from the previous revision on the
same host, because the machine is shared and a single run varies by roughly
15 percent.

| case | mojo-blosc2 | upstream blosc2 | relative | before | before relative |
| --- | ---: | ---: | ---: | ---: | ---: |
| compress int32 arange, 8 MB | 1.85 ms | 2.23 ms | 1.21x | 1.96 ms | 1.12x |
| decompress int32 arange, 8 MB | 1.33 ms | 1.17 ms | 0.88x | 4.24 ms | 0.31x |
| compress smooth float64, 16 MB | 17.28 ms | 12.16 ms | 0.70x | 18.12 ms | 0.61x |
| decompress smooth float64, 16 MB | 8.16 ms | 4.04 ms | 0.49x | 10.30 ms | 0.38x |
| compress random bytes, 8 MiB | 2.82 ms | 2.43 ms | 0.86x | 2.73 ms | 0.85x |
| decompress memcpy chunk, 8 MiB | 0.78 ms | 0.66 ms | 0.85x | 0.76 ms | 0.83x |

The decoder is where the port was furthest behind, and the two changes that
moved it are both about overlapping match copies rather than about the vector
width. Parsing upstream's int32 chunk shows 7,814 matches, every one of them
with a one-byte offset and an average length of 516 bytes, which the old
decoder ran one byte at a time; those now go through a broadcast 64-bit word
(`byte * 0x0101010101010101`, or a 16-bit seed for offset two) stored eight
bytes at a time. Offsets from eight to 31, which the old code only handled
byte at a time unless the offset divided the 32-byte block, now copy eight
bytes at a time.
The encoder's match extension had the same shape of problem: a 32-byte SIMD
compare followed by up to 31 scalar byte comparisons, now 32 bytes, then
eight, then four, then bytes. Its hash-table mode is split into two
compile-time specializations so the generation-tag test leaves the
inner loop, and the 4-byte word that feeds both the hash and the match check
is loaded once instead of twice.

The byte transpose was left alone. `shuffle_elements` and
`unshuffle_elements` already move 32 bytes per iteration and are bound by the
AVX2 shuffle and store ports rather than by loop overhead; they measure
3.48 ms and 2.17 ms for 16 MB of float64 in this revision, unchanged.
`compress` additionally skips the filter entirely when `typesize` is
one, where byte shuffle is the identity, which is not visible in the table
above because the benchmark selects NOFILTER for that case; it takes the
same call from 6.59 ms to 2.70 ms.

For the thresholded direct helpers, a 64 MB int32 buffer shuffles in 45.36 ms
and unshuffles in 39.00 ms. Smaller inputs remain serial.

There is intentionally no GPU path. The `std.gpu` module, including
`DeviceContext`, is absent from this toolchain, so no device path can be built
or run here. It would not pay off anyway: shuffle, fill, copies, and
hash-table clearing move far more bytes than arithmetic operations, while LZ4
parsing is branch-dependent and sequential within each stream. None reaches
the roughly two-flops-per-byte threshold where device transfer and launch
costs would be justified.

There is also no CPU parallelism. `std.algorithm.parallelize`,
`std.sys.parallelize`, and `std.sys.spawn` are all absent from this
toolchain, and `std.algorithm` no longer exports `sort`, so the work is
serial. The `nthreads` argument still splits a shuffle into independent
element ranges at 32 MiB and above, but those ranges run one after another on
the calling thread, so `nthreads=4` and `nthreads=1` measure the same.

Two further candidates were measured and rejected, and the code was left as it
was rather than made more complex for the same number:

- Widening the `typesize=8` shuffle from four-byte plane stores to eight-byte
  stores, by loading two 32-byte vectors and joining the corresponding byte
  planes. The shuffle kernel is bound by the AVX2 shuffle port, and the join
  costs as much port-5 work as the narrow stores cost port-4 work: 3.28 ms
  became 3.99 ms.
- Widening the tail of the shared `copy_bytes` helper to eight- and four-byte
  steps. The benchmark could not resolve a difference, so the simpler
  original was kept.

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
