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
above, but there is no worker API on this toolchain, so those ranges still run
one after another on the calling thread and the split measures no faster than
the serial loop (see Parallelism below).
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
faster.

These are the medians of ten `pixi run bench` runs on a shared host (72 cores,
load average between about 17 and 49 depending on other tenants, and a shared
RTX 5090 that ranged from idle to 100 percent GPU utilization from unrelated
work), measured 2026-11-11. Eight of the ten runs were interleaved with other
commands; the Rows section of `bench/verify.md` gives every individual reading,
which agree with these medians within 20 percent except for isolated outliers
on the random-bytes and memcpy rows. The `before` column is the same case
measured from `f886e3e`, built from source on the same host in the same
session; both revisions measured level with each other on every row. That
revision is an ancestor, not the previous README revision, and the two columns
do not measure the same thing.

| case | mojo-blosc2 | upstream blosc2 | relative | before | before relative |
| --- | ---: | ---: | ---: | ---: | ---: |
| compress int32 arange, 8 MB | 1.88 ms | 2.19 ms | 1.16x | 2.10 ms | 1.12x |
| decompress int32 arange, 8 MB | 1.54 ms | 1.31 ms | 0.85x | 1.54 ms | 0.85x |
| compress smooth float64, 16 MB | 17.95 ms | 12.98 ms | 0.72x | 17.94 ms | 0.72x |
| decompress smooth float64, 16 MB | 9.22 ms | 5.35 ms | 0.58x | 9.22 ms | 0.58x |
| compress random bytes, 8 MiB | 2.93 ms | 2.31 ms | 0.79x | 3.11 ms | 0.82x |
| decompress memcpy chunk, 8 MiB | 0.87 ms | 0.72 ms | 0.83x | 0.94 ms | 0.83x |

Mojo remains slower than upstream on five of the six rows and the table says so.
Two relative figures moved by 20 percent or more from the previous README and
are reported here as measured: `decompress smooth float64` (0.49x to 0.58x) and
`compress random bytes` (0.87x to 0.79x). The other four relative figures moved
by at most 0.04. `bench/verify.md` lists every individual reading behind each
row.

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
AVX2 shuffle and store ports rather than by loop overhead; measured
2026-11-11 they take 3.63 ms and 2.32 ms for 16 MB of float64 through
`byte_shuffle` and `byte_unshuffle` (the 2.17 ms figure previously quoted here
did not reproduce; see `bench/verify.md`).
`compress` additionally skips the filter entirely when `typesize` is
one, where byte shuffle is the identity. That path is not covered by a
reproducible timing here: the same 8 MiB call measures 2.68 ms with NOFILTER
and 2.61 ms with SHUFFLE, and a single earlier sample of that pair under a
loaded host read 6.59 ms and 2.70 ms, so the saving is not claimable at 2x.

For the thresholded direct helpers, a 64 MB int32 buffer shuffles in 57.52 ms
and unshuffles in 50.90 ms at `nthreads=1`. The 45.36 ms and 39.00 ms quoted
here previously did not reproduce. Smaller inputs remain serial; note that
above the split, `nthreads > 1` currently deduplicates the last element plane
per worker (see `bench/verify.md`), so the threshold table below reports the
serial figures.

There is intentionally no GPU path. **The implementation is CPU only, and on a
quiet host it is memory-bound.** On this shared host it was also measured while
the GPU was saturated by other tenants (see `bench/verify.md`), which makes
CPU-only timings worse here but is not a property of the code.

The gate for a GPU phase is a kernel above roughly two flops per byte, measured
in `bench/analysis.md`. The highest figure anywhere in that table is 0.51
(`decompress2` int32 / LZ4 decode int32) and the dominant stages sit at
0.06-0.32; shuffle, fill, copies, and hash-table clearing move far more bytes
than arithmetic operations, and LZ4 parsing is branch-dependent and sequential
within each stream. Every hotspot is therefore below the threshold, so no
kernel becomes a GPU candidate and none is worth a 16 MB device transfer per
call. That verdict rests on the measurement, not on the toolchain.

The toolchain closes the door as well. The `std.gpu` module, including
`DeviceContext`, is absent from this pinned toolchain, so no device path can be
built or run here. For the record, the shared GPU was checked as the phase
required: `nvidia-smi --query-gpu=memory.free` reported 5552 MiB free at the
time, above the 4000 MiB floor, so the skip is the flops-per-byte gate, not
memory pressure. No GPU code, test, or benchmark was added; there are
therefore no GPU tests to skip, on this host or a CPU-only one.

There is also no CPU parallelism. `std.algorithm.parallelize`,
`std.sys.parallelize`, and `std.sys.spawn` are all absent from this
toolchain, and `std.algorithm` no longer exports `sort`, so the work is
serial. The `nthreads` argument still splits a shuffle into independent
element ranges at 32 MiB and above, but those ranges run one after another on
the calling thread. The split is output-preserving: it was checked byte for
byte against the serial path at 32, 33, 40, 48 and 64 MiB (see the split
shuffle section below).

### Parallelism: what was measured, and why no parallel path shipped

The mandated API, `parallelize` from `max.algorithm`, cannot be compiled here.
The pinned environment (`mojo ==1.2.0.dev2026092605`) ships no `max` package at
all — `pixi run mojo build` fails with `unable to locate module 'max'` on
`from max.algorithm import parallelize`, and the Python side has no `max` module
either. Probing every module that does exist gives the same answer as
`MOJO_NOTES.md` section 4: `std.algorithm` has no `parallelize`, no `spawn`,
no `parallel_for`; `std.sys` has none either; `std.threading` does not exist.
No worker-creation primitive is reachable, so a real multi-core split could not
be written, let alone benchmarked. Nothing in `src/` was changed in this pass:
a parallel path that cannot be built would be a claim, not a change.

What was measured instead is the split that already exists in the codebase,
`shuffle_bytes` / `unshuffle_bytes` dividing the element range across
`workers` (driven by `nthreads`), against `nthreads=1`. Best of five timed
runs after one warmup, `uint32` input, `byte_shuffle` / `byte_unshuffle`
directly, remeasured 2026-11-11:

| bytes | op | 1 thread | 2 | 4 | 8 | 4 vs 1 | same output |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 4 MiB | shuffle | 0.59 ms | 0.61 | 0.59 | 0.59 | 1.01x | yes |
| 4 MiB | unshuffle | 0.35 ms | 0.36 | 0.35 | 0.36 | 1.00x | yes |
| 8 MiB | shuffle | 1.19 ms | 1.21 | 1.15 | 1.20 | 1.03x | yes |
| 8 MiB | unshuffle | 0.76 ms | 0.72 | 0.75 | 0.72 | 1.01x | yes |
| 16 MiB | shuffle | 2.74 ms | 2.76 | 2.80 | 2.71 | 0.98x | yes |
| 16 MiB | unshuffle | 1.92 ms | 2.00 | 1.82 | 1.83 | 1.05x | yes |
| 32 MiB | shuffle | 28.28 ms | 27.37 | 27.25 | 27.46 | 1.04x | yes |
| 32 MiB | unshuffle | 24.31 ms | 23.85 | 23.48 | 23.71 | 1.04x | yes |
| 64 MiB | shuffle | 55.12 ms | 54.80 | 56.27 | 56.58 | 0.98x | yes |
| 64 MiB | unshuffle | 50.78 ms | 50.62 | 49.52 | 64.43 | 1.03x | yes |
| 128 MiB | shuffle | 116.69 ms | 135.81 | 131.04 | 138.76 | 0.89x | yes |
| 128 MiB | unshuffle | 130.98 ms | 141.03 | 188.06 | 150.03 | 0.70x | yes |

Every ratio from 4 to 64 MiB is between 0.98x and 1.05x with no trend in
thread count, which is what a serial loop looks like. There is no size at
which the split wins, so there is no threshold to derive, and the existing
32 MiB `PARALLEL_THRESHOLD` is not supported by any speedup. At 128 MiB the
splits look slower, but this host's load average rose from 17 to 49 during
these measurements, so those rows are not used as evidence for or against a
split; `bench/verify.md` records the earlier, quieter and contradictory
readings.

The caller path agrees. `compress2` / `decompress2` on 32 MB float64 and 16-64 MB
int32, `nthreads` 1 / 2 / 4 / 8, best of five:

| case | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| compress smooth float64, 32 MB | 36.75 ms | 40.71 | 39.23 | 37.46 |
| compress int32 arange, 16 MB | 3.76 ms | 3.61 | 3.55 | 3.57 |
| compress int32 arange, 64 MB | 18.92 ms | 18.75 | 17.38 | 20.61 |
| decompress smooth float64, 32 MB | 18.31 ms | 18.37 | 18.45 | 18.36 |

Compressed bytes were asserted identical across all four thread counts, and
decompressed bytes identical to the source, inside the harness. The only
differences are host noise; `nthreads` is a no-op for performance. The earlier
version of this table carried much larger absolute figures (for example 524 ms
for the 32 MB float64 compress); they did not reproduce, and the relative
pattern is the same.

So the outcome of this pass is a revert/no-ship, stated plainly: no parallel
path was added, the existing split was not extended or tuned, and no threshold
was assigned from these numbers because no thread count above one won at any
size. Real callers here default to `nthreads=1` anyway, and the sizes where a
split could theoretically pay (>= 32 MiB) are also where it measures neutral to
slower. The performance table above is therefore unchanged by this pass: there
are no before/after numbers to report because there is no after.

A loop-tail and unrolling reduction pass was run against this path and kept
nothing, so
the table above is one revision against itself. What was tried, and the
measurement that rejected it:

- Multi-accumulator reductions were not applicable. There is no
  floating-point accumulation anywhere in the codec: the hot loops are byte
  gathers, a hash table, and match copies. Nothing was reordered, and the
  parity tests, which include every handcrafted decoder case and bidirectional
  upstream decoding, pass unchanged with no tolerance touched.
- Compile-time unrolling of the fixed-trip-count loops. This toolchain has
  replaced `@parameter for` with `comptime for`, and `@parameter` on a `for`
  is now a parse error. The one loop whose trip count is a true constant, the
  scalar byte-plane gather in `shuffle_elements`, was rewritten as a
  `comptime for byte_index` with `typesize` as a compile-time parameter and
  moved into its own helper. It measured 115.0 us per 256 KiB `typesize=4`
  block against 113.8 us for the plain `for` loop, inside the host's noise, so
  it was reverted rather than kept for a more complex kernel.
- The other small fixed-count loops already compile to their unrolled form.
  `read_u32` emits one 32-bit `mov`, and `write_u32`/`write_u64` collapse into
  a single store, so there is no loop left to remove.
- Removing bounds-check and index arithmetic in `match_length`. After the
  32-byte, 8-byte and 4-byte stages the remaining at most three bytes are
  scalar on purpose: a short tail amortises to under 0.2 percent of a
  compress. Two wider rewrites of that tail were built and rejected on
  correctness before timing, a byte-equality mask on a 4-byte xor that
  counted matches wrongly (int32 output 128,620 bytes instead of 89,705) and
  a `count_leading_zeros` prefix count that failed the parity tests.

Two further candidates from an earlier revision were also measured and
rejected, and the code was left as it was rather than made more complex for
the same number:

- Widening the `typesize=8` shuffle from four-byte plane stores to eight-byte
  stores, by loading two 32-byte vectors and joining the corresponding byte
  planes. The shuffle kernel is bound by the AVX2 shuffle port, and the join
  costs as much port-5 work as the narrow stores cost port-4 work: 3.28 ms
  became 3.99 ms.
- Widening the tail of the shared `copy_bytes` helper to eight- and four-byte
  steps. The benchmark could not resolve a difference, so the simpler
  original was kept.

## Split shuffle: what was checked

`shuffle_elements` and `unshuffle_elements` accept a `start`/`stop` element
range, and the split calls each worker with the whole element count plus its own
`start`/`stop`. I read this code as being able to write outside a worker's
range, and wrote a reproduction around that reading. It did not reproduce, so
the reading is wrong and the kernel is treated as correct here.

What was checked on 2026-11-11: `uint32` inputs of exactly 32, 33, 40, 48 and
64 MiB, `byte_shuffle` and `byte_unshuffle` at `nthreads` 1, 2, 4 and 8,
compared byte for byte with each other and with a pure-Python reference
implementation, plus round-trips back to the source. Every comparison matched
with zero differing bytes. `pixi run test` covers the same property at 4 MiB,
32 MiB and `(32 MiB) + 12` for `typesize` 4 and 8. The claim in the earlier
README and in an earlier draft of this section that the split corrupts output
was not demonstrated and is not restated.

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
