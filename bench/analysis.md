# ANALYZE: where the time goes in mojo-blosc2

Codec path of `src/blosc2.mojo` as pinned by `MOJO_NOTES.md` (Mojo
`1.2.0.dev2026092605`, AVX2 / 32-byte vectors). No algorithm was changed in this
phase and nothing under `src/` or `tests/` was touched.

Everything below was measured in this session on the host that `pixi run bench`
reports. Harnesses live outside the repo; the only in-repo change is this file.

## Method

| step | command | notes |
| --- | --- | --- |
| required bench | `pixi run bench` | flock-wrapped, run twice (first and last) |
| diagnostic variant | `MOJO_BENCH_PROFILE=1 pixi run bench` | adds shuffle/thread rows |
| session sweeps | `flock /tmp/mojo-bench.lock python /tmp/phaseA_*.py` | same lock, never concurrent |

- Machine: `Intel(R) Xeon(R) CPU E5-2697 v4 @ 2.30GHz; Linux x86_64; Python 3.13.14`.
  L1d 32 KiB/core, L2 256 KiB/core, L3 45 MiB/socket, AVX2.
- Upstream comparison: `blosc2 4.9.1`, `nthreads=1`, LZ4, 256 KiB blocks,
  clevel 5, identical filter list, on the identical input arrays.
- Baseline for the byte-stream kernels is `bytes(memoryview)` on the same
  array (a verbatim block copy). For the plane kernels it is a NumPy plane
  shuffle (`np.concatenate([b[:, j].copy() ...])`), because an element copy is
  the wrong reference for a transpose.
- Each value is the **best of 7** runs after one warmup, matching the bench. The
  host is shared: individual timings move by 10-20 % between runs. Where that
  matters (shuffle block size) a 31-repetition min/median is quoted and labelled
  `n31`. Times are milliseconds.
- "In kernel" means the cost is inside the Mojo `.so`. "Glue" means it is in
  `python/mojo_blosc2` before or after the single exported C call.
- Attribution (`Δfilter`) is measured, not modelled: `compress2(SHUFFLE)` minus
  `compress2(NOFILTER)` on the same array. That isolates the filter plus the
  LZ4 work the filtered stream causes, and it can be negative when shuffle turns
  LZ4-heavy data into cheap runs.
- `flops` counts compare/subtract/gather steps only; there is no floating-point
  arithmetic anywhere in this path. That is what drives the accelerator verdicts
  at the end.
- Rows marked **slower** are slower than upstream on that input. None were
  dropped.

Inputs, as constructed in `bench/bench.py`:

| input | bytes | chunk bytes | blocks | upstream ratio |
| --- | ---: | ---: | ---: | ---: |
| `int32 arange(2e6)` + SHUFFLE | 8,000,000 | 89,705 | 31 (`262144`) | 142.2x upstream (56,248) |
| `float64 sin(linspace)` + SHUFFLE | 16,000,000 | 11,861,547 | 62 (`262144`) | 1.37x |
| `uint8 random` + NOFILTER | 8,388,608 | memcpy chunk | 32 | 1.00x |

## Ranked hotspots

Ranked by `ms`. Sizes are those in `bench/bench.py` plus the small and 64 MB
sweeps, i.e. what real callers pass today, from 64 B `pack` payloads to 64 MB
direct shuffles. `GB/s` is `bytes touched / time` and `Gflop/s` is
`flops / time`. Δfilter is in the notes column. "excerpted copy" and "by
subtraction" say how that row was obtained when no single public call covers it.

| # | function | input size | time (ms) | bytes touched | flops (approx) | flops/byte | bound by | glue or kernel | note |
| --: | --- | --- | --: | --: | --: | --: | --- | --- | --- |
| 1 | `compress2` float64 SHUFFLE (**slower**) | 16,000,000 | 17.48 | 43.6 M | 12.5 M | 0.29 | compute (LZ4 branches) | kernel | Δfilter +10.87 over NOFILTER 6.61; upstream 12.29 |
| 2 | `decompress2` float64 lz4+split (**slower**) | 16,000,000 | 7.99 | 35.9 M | 11.0 M | 0.31 | compute (LZ4 branches) | kernel | upstream 4.04; unshuffle alone 2.22 |
| 3 | LZ4 encode, float64 (in-chunk) | 16,000,000 | 14.08 | 32.0 M | 3.0 M | 0.09 | compute (branch/parse) | kernel (excerpted copy) | 1.14 GB/s; upstream's share ~9.0 (by subtraction) |
| 4 | LZ4 decode, float64 (in-chunk) | 16,000,000 | 5.77 | 27.9 M | 12.0 M | 0.43 | compute (branch/match copy) | kernel (by subtraction) | 8.40 total − 2.22 unshuffle − 0.18 glue |
| 5 | `byte_shuffle` float64, stream | 16,000,000 | 3.52 | 32.0 M | 2.0 M | 0.06 | memory (byte-store port) | kernel | 4.55 GB/s; copy ref 1.08 |
| 6 | `compress2` int32 SHUFFLE | 8,000,000 | 1.82 | 17.7 M | 5.6 M | 0.32 | compute (LZ4 branches) | kernel | Δfilter −0.90 (shuffle helps); upstream 2.24 |
| 7 | `compress2` float64 NOFILTER (**slower**) | 16,000,000 | 6.61 | 33.9 M | 3.0 M | 0.09 | compute (branch/parse) | kernel | upstream 5.02; 1.32x slower |
| 8 | LZ4 encode, int32 (in-chunk) | 8,000,000 | 4.06 | 16.0 M | 1.5 M | 0.09 | compute (branch/parse) | kernel (excerpted copy) | 1.97 GB/s |
| 9 | LZ4 decode, int32 (in-chunk) | 8,000,000 | 1.10 | 7.9 M | 4.1 M | 0.51 | compute (branch/match copy) | kernel (by subtraction) | 1.24 observable − 0.92 shuffle − 0.18 glue |
| 10 | `decompress2` int32 (**slower**) | 8,000,000 | 1.24 | 7.9 M | 4.1 M | 0.51 | compute (LZ4 branches) | kernel | upstream 1.12; 1.11x slower |
| 11 | `decompress2` memcpy chunk (**slower**) | 8,388,608 | 0.71 | 16.8 M | 4.2 M | 0.25 | memory | kernel | pure copy, 11.8 GB/s vs 13.4 for `bytes()`; upstream 0.59 |
| 12 | `compress2` uint8 random (**slower**) | 8,388,608 | 2.60 | 16.8 M | 4.2 M | 0.25 | compute (hash loop on noise) | kernel | upstream 2.22; 1.17x slower |
| 13 | `compress2` float64 random NOFILTER (**slower**) | 16,000,000 | 5.95 | 32.1 M | 3.0 M | 0.09 | compute (branch/parse) | kernel | upstream 4.87; ratio 1.05 so nothing to skip |
| 14 | `byte_shuffle` int32, stream | 8,000,000 | 0.92 | 16.0 M | 2.0 M | 0.13 | memory (byte-store port) | kernel | 8.65 GB/s; copy ref 0.56 |
| 15 | `decompress2` float64 lz4+unsplit | 16,000,000 | 7.42 | 35.9 M | 10.6 M | 0.30 | compute (LZ4 branches) | kernel | upstream 4.55 |
| 16 | `byte_unshuffle` float64, stream | 16,000,000 | 2.22 | 32.0 M | 2.0 M | 0.06 | memory (byte-store port) | kernel | 7.2 GB/s |
| 17 | `byte_unshuffle` int32, stream | 8,000,000 | 0.67 | 16.0 M | 2.0 M | 0.13 | memory (byte-store port) | kernel | 11.9 GB/s |
| 18 | `compress2` int32 NOFILTER (**slower**) | 8,000,000 | 2.71 | 17.7 M | 2.6 M | 0.15 | compute (hash/parse on raw bytes) | kernel | upstream 2.23; 1.22x slower; loses the 0.90 ms SHUFFLE saves |
| 19 | `decompress2` int32 all-zero chunk | 8,000,000 | 0.36 | 8.0 M | 0.5 M | 0.06 | memory | kernel | upstream 0.35 |
| 20 | `compress2` / `decompress2` glue @ 64 B | 64 | 0.039 / 0.012 | 0.1 K | ~0 | — | glue (fixed cost) | glue | upstream 0.0138 / 0.0028; fixed floor 0.030-0.045 |
| 21 | `pack` int32 1000 elems | 4,000 | 0.046 | 8 K | ~0 | — | glue (pickle + fixed) | glue | upstream 0.053 |

## Time splits that matter

Compress, float64 (input 16 MB, 256 KiB blocks, clevel 5):

| stage | ms | share | how measured |
| --- | --: | --: | --- |
| shuffle of input | 3.52 | 20 % | `byte_shuffle` on the same buffer |
| LZ4 encode (62 blocks x 8 split streams) | 14.08 | 80 % | excerpted copy of `lz4_compress` over the shuffled buffer |
| Python glue (hash checks, buffer export, header) | ~0.18 | 1 % | `decompress2` glue floor, measured below |
| total | 17.48 | | `compress2` |

The excerpted-copy sum is 17.78 against a 17.48 total measured 25 s earlier on
the same host, so the two attributions agree within the host's 15 % noise; the
split is 20 / 80, not 25 / 75. Compressing the shuffled float64 stream as
**one** 16 MB LZ4 block instead of 62x256 KiB costs 483.7 ms: blocks are
already the right granularity.

Compress, int32 (input 8 MB): shuffle 0.92 ms (52 %), LZ4 encode 4.06 ms (far
more than the 1.82 measured total because shuffle converts this data into
runs/1-byte-offset matches that LZ4 encodes almost for free). Measured
attribution: `Δfilter = SHUFFLE − NOFILTER = -0.90 ms`.

Decompress, float64: LZ4 decode 5.77 ms (72 %), unshuffle 2.22 ms (28 %),
glue 0.18 ms (2 %).

Why float64 is the slow case: `float64 sin(...)` compresses only 1.37x, so the
encoder spins through 62 blocks x 8 streams of mostly-incompressible bytes
(1.14 GB/s), whereas upstream manages about 9.0 ms for the same LZ4 work.
int32 `arange` compresses 142x, so the encoder exits early on runs.

## Why the LZ4 encoder is the top hotspot (evidence)

| probe | result | reading |
| --- | --- | --- |
| `compress2` float64 shuffle at clevel 1 / 5 / 9 | 14.63 / 18.01 / 26.65 ms | 1.83x cost from clevel 1 to 9 |
| upstream, same sweep | 11.14 / 12.69 / 16.85 ms | upstream: 1.51x; Mojo starts 31 % behind and ends 58 % behind |
| upstream splitmode auto / never / always, SHUFFLE | 13.26 / 12.27 / 13.63 ms | split is not the explanation |
| Mojo NOFILTER vs upstream NOFILTER, float64 | 6.61 vs 5.02 ms | LZ4-only path is 1.32x slower, no shuffle involved |
| same roundtrip on random float64 (ratio 1.05) | Mojo 21.02 / upstream 12.24 ms SHUFFLE | 1.72x; the gap is in the parser, not the filter |

So the encoder's search loop (`read_u32` + hash + table probe + match extend)
costs 31-72 % more than upstream LZ4 regardless of data. The clevel slope says
the excess scales with search attempts, not with output bytes.

## Python/ctypes glue audit

- **No Python loop re-enters ctypes per element anywhere in the shipped code.**
  `python/mojo_blosc2` contains exactly one `for` loop (`_lib.py:106`), over four
  signature entries at library load. Every hot path makes **one** `lib().mbl_*`
  call: `_compress` (1), `_decompress_to` (1), `byte_shuffle` (1),
  `byte_unshuffle` (1).
- All element-wise iteration (blocks, split streams, shuffle planes, LZ4
  matches) is in Mojo. The inner `while` loops in `shuffle_elements` /
  `unshuffle_elements` / `lz4_encode` never cross the ABI.
- Reference point for what such a loop would cost: a Python-per-plane shuffle
  (`numpy_plane_shuffle`) takes 24.65 ms for 16 MB versus 3.52 ms in the kernel.
- Glue floor, measured on paths with no user work: `compress2`/`decompress2` on
  64 B = 0.039 / 0.012 ms; `decompress2` on a 32-byte all-zero chunk = 0.012 ms.
  Upstream's floor is 0.0138 / 0.0028 ms, so glue is 0.008-0.025 ms; the upper
  end is the 0.025 ms difference against upstream at 64 B. At 16 MB that is under
  0.3 % of decompress; only at <= 1 KiB does it dominate (Table row 20).
- `nthreads=2/4` adds no cost and no speedup (`shuffle 64 MB`: 49.77 / 49.87 /
  48.80 ms). Confirms `MOJO_NOTES.md` §4: the `workers` split in
  `shuffle_bytes` is a serial `for item` loop, not parallelism.
- `byte_shuffle(typesize=1)` is an identity copy of the whole buffer, not a
  no-op: `ts1` measures 7.78 ms for 16 MB, which is exactly the per-block
  serial plane loop. `compress2` already skips it.

## Per-call allocations and copies

| site | allocation/copy | size | cost |
| --- | --- | --- | --- |
| `core._compress` (via `writable_bytes`) | `PyBytes_FromStringAndSize(N+32)` | N+32 | runtime alloc, zero-fill |
| `core._compress` return (`destination[:result]`) | second full copy, then the N+32 buffer is freed | cbytes | `int32 8 MB` -> 89,705 B; `float64 16 MB` -> 11.86 MB |
| `core._decompress` (fresh bytes) | `PyBytes_FromStringAndSize(nbytes)` | N | + `dst=` variant shows no saving (1.20 vs 1.18 ms), so this is not a hotspot |
| `_parse_header` | `ctypes.string_at(addr, 32)` -> `bytes`, twice per decompress (line 178 and line 249) | 32 B | inside the 0.012 ms floor |
| `_lib.BufferExport.__init__` | `_PyBuffer` + `memoryview().cast("B")` per call | — | inside the 0.012 ms floor |
| `_lib.workspaces` | `bytearray(blocksize)` + `BufferExport` + `c_int32*65536` once per thread (reused after) | ~256 KiB | amortised to ~0 |
| kernel | none. Mojo allocates nothing; scratch/table are caller-owned | — | per README |

The single full-length extra copy on the compress return is 11.86 MB for the
float64 case; it sits inside the measured total and is negligible next to the
14 ms encoder. No copy is large enough to be a hotspot on its own.

## Shuffle kernel: measured bottleneck, and one host artifact

| probe | result | reading |
| --- | --- | --- |
| `byte_shuffle` int32 at 1/4/8/16 MB | 6.71 / 7.05 / 7.22 / 6.20 GB/s | flat: not memory-bandwidth bound |
| `byte_shuffle` int32 32 / 64 MB | 1.25 / 1.26 GB/s | cliff at 32 MB, exactly `PARALLEL_THRESHOLD` |
| the same 32 MB shaped as 256 KiB blocks vs 4 KiB blocks | 52.67 vs 26.77 ms | the cliff is **block size**, not the thread branch: identical byte count, 2x apart |
| `byte_unshuffle` int32 64 MB, nthreads 1/2/4 | 46.10 / 46.34 / 45.51 ms | the `workers` split is serial |
| a single 256 KiB block, all planes, repeated | 0.0920 ms median (n31) | **2.85 GB/s** |
| the same block, 4 KiB or 16 KiB sub-blocks | 4.07-4.08 GB/s whole-buffer | faster than 256 KiB blocks |
| 64-byte store loops compiled in a scratch copy of the module | compiles, crashes at runtime | `SIMD[DType.uint8, 64]` on this AVX2 build |
| aligned vs +4 B shifted int32 view | 7.12 vs 6.94 GB/s | alignment is not the issue |

Two things follow.

1. A single 256 KiB block, which is L2-resident, still runs at only 2.85 GB/s.
   So the loop is not waiting on DRAM; it is spending its time in the plane-gather
   store sequence. 256 KiB blocks therefore cost 2x what 4 KiB sub-blocks do on
   the same bytes (52.67 vs 26.77 ms for 32 MB), and the block size the encoder
   uses (`256 KiB`) is on the wrong side of that cliff. The same effect shows up
   in `compress2` (block-size table above: 11 % on float64, 27 % on int32 at
   1 MiB), so it is measured end to end, not only in an excerpted kernel.
2. The store path is limited by one 32-byte store per plane per iteration
   (8 stores for `typesize=8`, 4 for `typesize=4`). All existing "make planes
   wider" attempts already failed (README: eight-byte plane stores
   3.28 -> 3.99 ms), so this phase makes no code claim beyond what is measured
   here: the realistic lever is fewer stores per element, not wider stores.

## Per-hotspot next phase

The threshold used for `gpu` is **>= 2 flops/byte**, the figure the README
states. No row in this table is close: the highest is 0.51, so no hotspot
qualifies, and `std.gpu` is absent from this toolchain anyway (MOJO_NOTES §5).
That verdict is based on the measured figure, not on the missing module.

| hotspot | flops/byte | verdict | reason from the measurements |
| --- | --: | --- | --- |
| LZ4 encode (rows 1, 3, 6, 7, 8, 12, 13, 18) | 0.09-0.32 | **unroll** | 14.08 of 17.48 ms; 1.31-1.72x behind upstream at every clevel; slope with clevel says per-search-attempt overhead. Sequential dependency, so not simd/parallel/GPU. Secondary, cheaper: see the block-size table below (6 % of float64 compress, 24 % of int32 compress at 16 KiB blocks). |
| LZ4 decode float64 (rows 2, 4) | 0.43 | **unroll** | 5.77 of 7.99 ms; match-copy and length-extension are the serial part; already partly widened, still 2x upstream. |
| shuttle stream, float64 (rows 5, 16) | 0.06 | **simd** | 3.52 / 2.22 ms; store-port limited (8 planes/iter), not cache-width limited (some wider variants already measured slower). Any change must be a new layout, not a wider store. Block size interacts: 256 KiB blocks cost 1.58 ms more than one pass for 16 MB. |
| shuffle stream, int32 (rows 14, 17) | 0.13 | none, keep | 0.92 / 0.67 ms and behind a copy only 1.7x/1.2x; already helps LZ4 (Δfilter −0.90 ms). |
| memcpy chunk decode (row 11) | 0.25 | **simd** | 11.8 vs 13.4 GB/s for a plain `bytes()` copy: the `copy_bytes` loop is 12 % off glibc `memcpy`. Small, cheap, bounded upside. |
| all-zero / run decode (row 19) | 0.06 | none | already at parity with upstream. |
| Python glue (rows 20, 21) | — | none | 0.008-0.025 ms fixed; 0.1-1 % of any input >= 256 KiB. Fixing it cannot move the table. |

## What the block size changes, measured

End to end, same data, only `blocksize` varies (min of 9; upstream alongside).
Every number here is a real `compress2` call.

| blocksize | float64 16 MB mojo / upstream | int32 8 MB mojo / upstream | int32 cbytes |
| --: | --: | --: | --: |
| 16 KiB | 15.65 / 13.06 | 1.687 / 2.585 | 189,680 |
| 64 KiB | 16.42 / 11.93 | 1.700 / 2.171 | 82,942 |
| 256 KiB (bench default) | 17.57 / 12.15 | 1.799 / 2.197 | 56,248 |
| 1 MiB | 22.46 / 13.49 | 2.281 / 2.285 | 57,905 |

- float64 at 16 KiB is 1.92 ms (11 %) faster than at 256 KiB, closing about a
  third of the 5.4 ms gap to upstream. At 1 MiB it is 4.9 ms slower. Upstream is
  flat across the sweep (11.9-13.5 ms), so this is a property of the port.
- int32 at 1 MiB has a **+27 %** jump (1.799 -> 2.281 ms, upstream flat at 2.28):
  the shuffle goes to 256 KiB-block DRAM speed at exactly the same time.
- The cost of smaller blocks: int32 cbytes grows 3.4x (56,248 -> 189,680).
  Block size trades ratio against time, and it is a caller-visible knob
  (`blocksize=`) and a default in `_auto_blocksize`, not an algorithm change.

**GPU:** none of the hotspots reach 2 flops/byte (maximum measured 0.51), and
the two dominant stages are sequential LZ4 parse/copy whose cost is branch
mispredicts, not arithmetic. A device transfer of 16 MB would move 32 M
bytes through PCIe for an instruction mix with 0.09 flops/byte of arithmetic.
Verdict: **no gpu phase for any hotspot.**

**parallel:** `MOJO_NOTES` §4 stands, and the measurements agree: `nthreads=4`
changes a 64 MB shuffle by -1.0 % (49.77 -> 48.80 ms), and 2 vs 1 thread on a
1 MiB-block compress is 23.02 vs 23.89 ms. No code path benefits today.
`nthreads` should be documented as a no-op contract, not as a performance knob.

## Fast paths worth knowing about before the next phase

| entry | size | mojo | upstream | baseline | verdict |
| --- | --- | --: | --: | --: | --- |
| `compress2` int32 SHUFFLE | 64 B | 0.039 | 0.014 | 0.0003 `bytes()` | glue floor, 2.8x |
| `compress2` int32 SHUFFLE | 1 KiB | 0.037 | 0.014 | 0.0004 | glue floor, 2.7x |
| `compress2` int32 SHUFFLE | 16 KiB | 0.046 | 0.019 | 0.0010 | 2.4x |
| `compress2` int32 SHUFFLE | 256 KiB | 0.100 | 0.081 | 0.017 | 1.2x slower |
| `compress2` int32 SHUFFLE | 1 MiB | 0.271 | 0.284 | — | at parity |
| `compress2` int32 SHUFFLE | 8 MiB | 1.82 | 2.24 | 0.56 `tobytes()` | 1.23x faster |
| `decompress2` int32 | 1 KiB | 0.014 | 0.004 | 0.0004 | glue floor, 3.7x |
| `decompress2` int32 | 256 KiB | 0.061 | 0.066 | 0.017 | at parity |
| `decompress2` int32 | 8 MiB | 1.24 | 1.12 | 0.54 | 1.11x slower |
| `pack` int32 1000 elems | 4,000 | 0.046 | 0.053 | — | at parity |
| `byte_shuffle` int32 | 8 MiB | 0.918 | — | 0.537 | 1.7x a plain copy, upstream has no direct helper |

The port is **faster than upstream on the highly compressible int32 path**
(1.17-1.18x) and **behind on the float64 path and on noise**. The crossover is
ratio, not size: at 142x compression the encoder exits early, so its slower
parser costs nothing; at 1.37x and 1.05x the parser is the whole job.

## Bench table, first and last run of the session

`pixi run bench`, first run then last run. The machine is shared, so the same
command moves by up to 15 %; both are shown rather than averaged.

| case | mojo first (ms) | upstream first (ms) | relative first | mojo last (ms) | upstream last (ms) | relative last |
| --- | --: | --: | --: | --: | --: | --: |
| compress int32 arange, 8 MB | 1.75 | 2.06 | 1.17x | 2.16 | 2.53 | 1.17x |
| decompress int32 arange, 8 MB | 1.29 | 1.10 | **0.85x** | 1.33 | 1.18 | **0.89x** |
| compress smooth float64, 16 MB | 16.57 | 12.37 | **0.75x** | 17.86 | 13.08 | **0.73x** |
| decompress smooth float64, 16 MB | 9.94 | 4.76 | **0.48x** | 9.14 | 5.52 | **0.60x** |
| compress random bytes, 8 MiB | 3.11 | 2.65 | **0.85x** | 2.94 | 2.32 | **0.79x** |
| decompress memcpy chunk, 8 MiB | 0.99 | 0.83 | **0.84x** | 0.97 | 0.74 | **0.76x** |

`MOJO_BENCH_PROFILE=1` (first session run): shuffle int32 8 MB 0.97 / unshuffle
0.76; shuffle float64 16 MB 3.82 / unshuffle 2.47; compress int32 NOFILTER 2.65;
decompress int32 NOFILTER 0.79; compress float64 1 MiB blocks 1 thread 23.89 /
2 threads 23.02; decompress 8.46 / 8.61; shuffle int32 64 MB 1 thread 47.50 /
4 threads 46.51; unshuffle 44.90 / 44.06.

## Caveats

- This host is shared (`README`: single-run variation about 15 %). Every
  conclusion above is stated as a range or direction that survives that noise;
  the `simd`/`unroll` verdicts rest on differences of 20-190 %, not 5 %.
- Host is shared, so the shuffle block-size figures were taken twice: 3 rounds of
  11 in `/tmp/phaseA_repeat.py` and the end-to-end `compress2` sweep in
  `/tmp/phaseA_blocksize.py` (9 reps, min and median printed). Only the direction
   and the >= 10 % effects are used as conclusions; the 2x block-size gap in the
   shuffle kernel and the 11 % / 27 % end-to-end effects each held across every
   repetition round.
- The LZ4-encode-only and plane-block numbers come from an **excerpted copy** of
  `src/blosc2.mojo` built outside the repo into `/tmp/streambench/libsb.so`. It
  starts as a byte-identical copy of the source and adds five exports
  (`sb_shuffle_stream`, `sb_unshuffle_stream`, `sb_lz4_only`, `sb_shuffle_blocks`,
  `sb_decode_only`); `sb_decode_only` stays unused. No existing function is
  modified. It is instrumentation, not a proposed change, and `src/` is
  untouched.
- One instrumentation attempt failed and is reported rather than hidden: a
  scratch 64-byte-store gather loop (`SIMD[DType.uint8, 64]`) compiles but
  crashes at runtime on this AVX2 build, so the 32 B / 64 B store-width
  comparison for a future `simd` phase is **not** available from this session.
  That is also a toolchain fact worth carrying into `MOJO_NOTES.md`.
- `flops` are instruction-count estimates from reading the kernels (one compare
  or subtract per element/byte in the plane loops, one hash + one compare per
  LZ4 probe, one byte per match-copy source byte). They are computed, not
  profiled, so `flops/byte` is an order-of-magnitude figure. The nearest hotspot
  (LZ4 decode, 0.51) is 4x below the 2.0 GPU threshold and the dominant ones are
  7x-22x below, far outside the uncertainty in that estimate. `bytes touched` is
  the row order, not a measurement.
