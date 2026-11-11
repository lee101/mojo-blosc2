# VERIFY: what in the README was reproduced, what was not, and what changed

This pass took the README's performance claims as unproven until re-measured.
Nothing under `src/` was changed. Files changed: `README.md` (numbers and the
claims that did not hold), `bench/bench.py` (timing-path leakage and per-run
output checks), `tests/test_parity.py` (one added parity test), and this file.

Session: 2026-11-11, `mojo-blosc2` at `8192ab6`, Mojo
`1.2.0.dev2026092605 (236c32cb)`, upstream `blosc2 4.9.1`, Python 3.13.14,
Intel Xeon E5-2697 v4 @ 2.30 GHz, 72 cores. The host is shared: load average
moved between about 17 and 49 during the session, and a shared RTX 5090 ranged
from idle to 100 percent GPU utilization from other tenants. Both affect
timings; every reading below is labelled.

## Gate

Run in order, at the end of the pass:

```
pixi run build   -> exit 0 (compiler deprecation warnings only)
pixi run test    -> 159 passed in 60.99s
pixi run bench   -> exit 0, table printed
```

Earlier in the session the untouched tree gave `pixi run build && pixi run test`
-> `155 passed in 50.05s`, exit 0. The 159 are the 155 plus the 4 new
`test_byte_shuffle_below_split_threshold_is_complete` cases.

A final re-run after the last `bench.py` edit, with the verified tree as
submitted:

```
pixi run build   -> exit 0
pixi run test    -> 159 passed in 62.79s
```

`pixi run bench` was run far more than once; the runs used for the README are
listed per row below.

## How the README numbers were taken

`pixi run bench` itself is unchanged in what it measures: both sides get the
same NumPy buffer, the same `codec/typesize/clevel/blocksize/nthreads/
splitmode/filters`, and decompression uses one upstream-produced chunk for both
sides. Each value is the best of 7 runs after 1 warmup, as the README says.
There is no size, thread count, or filter asymmetry between the two sides in
the harness.

Two changes were made to `bench.py`, neither of which changes which code runs
inside the timed call:

1. The per-run SHA-256 digest is now computed *after* the stopwatch stops. In
   the first version of this patch it was inside the timed region, and
   hashing 8 MiB per run inflated the `mojo` column (a 64 MiB shuffle went from
   26.35 ms to 32.94 ms and a 128 MiB one from 131 ms to 188 ms). That version
   was discarded and its numbers are not used anywhere.
2. Per-run output checks were added and are described under "Unchecked work".

The README table is the median of ten `pixi run bench` runs. Eight of those ten
were interleaved with other commands (that is why the host state moves between
them); the honest description is "ten runs on a busy shared host", not "ten
back-to-back clean runs". Individual readings follow.

## Rows

Each row: every individual `pixi run bench` reading used, then the median.

### compress int32 arange, 8 MB

mojo (ms): 2.15, 1.92, 1.78, 1.80, 2.13, 2.14, 1.77, 1.88, 2.10, 1.83, 1.79
median 1.88
upstream (ms): 3.18, 2.14, 2.17, 2.17, 2.48, 2.52, 2.19, 2.19, 2.54, 2.20, 2.14
median 2.19
relative 1.16x (README before: 1.19x, inside 20 percent; corrected to measured)

### decompress int32 arange, 8 MB

mojo: 1.71, 1.38, 1.53, 1.50, 1.34, 1.64, 1.54, 1.61, 1.52, 1.92, 1.53
median 1.54
upstream: 1.37, 1.25, 1.29, 1.29, 1.12, 1.37, 1.33, 1.34, 1.32, 1.21, 1.39
median 1.31
relative 0.85x (README before: 0.83x). Individual mojo runs reach 1.92
(+25 percent over the median) on the noisy host; the relative stayed within
20 percent on every pairing except that one.

### compress smooth float64, 16 MB

mojo: 18.53, 17.96, 17.20, 17.30, 16.68, 17.94, 22.48, 17.33, 21.20, 18.40, 18.17
median 17.95
upstream: 12.19, 13.12, 12.47, 12.50, 11.61, 13.32, 15.39, 13.07, 12.89, 14.15, 13.47
median 12.98
relative 0.72x (README before: 0.70x)

### decompress smooth float64, 16 MB

mojo: 10.29, 10.41, 8.55, 9.20, 8.52, 10.63, 9.22, 10.98, 9.05, 9.05, 8.69
median 9.22
upstream: 4.93, 5.65, 4.42, 4.58, 5.35, 5.44, 5.06, 5.11, 5.71, 5.71, 4.39
median 5.35
relative 0.58x (README before: 0.49x, a 21 percent relative change; corrected to
the measured 0.58x)

### compress random bytes, 8 MiB

mojo: 2.82, 3.19, 2.98, 2.95, 2.77, 2.83, 2.79, 2.77, 3.75, 3.11, 3.43, 2.93, 2.68
median 2.93
upstream: 2.26, 2.64, 2.30, 2.31, 2.22, 2.31, 2.25, 2.22, 2.56, 2.75, 2.79, 2.52, 3.79
median 2.31
relative 0.79x (README before: 0.87x, a 9 percent change; corrected to the
measured value). The 3.79 ms upstream reading is an outlier from the final busy
run; it moved the median by 0.01 ms.

### decompress memcpy chunk, 8 MiB

mojo: 0.74, 0.93, 0.78, 0.80, 0.94, 1.12, 0.91, 0.80, 0.96, 0.89
median 0.87
upstream: 0.63, 0.81, 0.65, 0.65, 1.13, 1.20, 0.78, 0.67, 0.72, 0.73
median 0.72
relative 0.83x (README before: 0.81x). This row is 0.65-1.2 ms; the host's
scheduling noise is a larger fraction of it than of any other row.

### Rows the README says are slower

Five of six rows are below 1.00x and stay in the table, unreworded. No row was
removed.

## The "before" column

The README described `before` as "the previous revision (`f886e3e`) built from
source and benched on the same host immediately afterwards". I built `f886e3e`
in a `git worktree` at `/tmp/wt-f886e3e` and ran `pixi run bench` four times:

- compress int32: 2.18, 1.84, 1.90, 1.80 -> median 1.87 ms
- decompress int32: 1.65, 1.31, 1.45, 1.88 -> median 1.55 ms
- compress smooth float64: 21.67, 17.23, 18.04, 17.02 -> median 17.63 ms
- decompress smooth float64: 10.71, 8.60, 9.52, 10.07 -> median 9.79 ms
- compress random bytes: 3.54, 3.70, 2.91, 2.78 -> median 3.23 ms
- decompress memcpy: 1.01, 0.86, 0.89, 1.02 -> median 0.95 ms

The random-bytes `before` in the README was 3.07 ms; four fresh runs of the
same revision medians 3.23 ms, so the column is corrected. With the corrected
values, `f886e3e` and `HEAD` measure level on every row, which agrees with the
README's statement that the intervening pass kept no source change. The column
is also now described as an ancestor revision, not the previous README
revision; the README previously implied the latter.

## Other README numbers re-measured

| README claim | measured 2026-11-11 | status |
| --- | --- | --- |
| `shuffle_elements` 3.48 ms / `unshuffle` 2.17 ms for 16 MB float64 | 3.63 ms / 2.32 ms (5 rounds x min-of-5, final run) | corrected |
| direct helper, 64 MB int32 shuffle 45.36 ms / unshuffle 39.00 ms | 57.52 ms / 50.90 ms | corrected |
| `typesize=1` NOFILTER "6.59 ms" to `typesize=1` SHUFFLE "2.70 ms" | 2.68 ms NOFILTER, 2.61 ms SHUFFLE, same 8 MiB call | claim withdrawn; the 6.59 ms reading could not be reproduced and is not a 2x saving here |
| parallel split table (4-128 MiB, 1/2/4/8 threads) | remeasured; every ratio 0.98x-1.05x from 4-64 MiB; 128 MiB rows landed at 0.70x-0.89x while the host load hit 49 | table replaced with measured values; 128 MiB rows labelled host-load-affected rather than cited as a split speedup or slowdown |
| caller `nthreads` table (32 MB float64 compress 524.48 ms etc.) | 36.75 / 40.71 / 39.23 / 37.46 ms for 1/2/4/8 | replaced; relative pattern is the same, absolute figures did not reproduce |
| GPU flops-per-byte gate, highest 0.51 from `decompress2` int32 | `bench/analysis.md` row 10, time 1.24 ms, upstream 1.12; a fresh `decompress2` of the same chunk on this host takes 2.19-2.45 ms | 0.51 itself is not reproducible as a stable figure with today's `decompress2` timings; the conclusion (all hotspots below 2 flops/byte) is unchanged. `bench/analysis.md` was left alone because rewriting it is outside this pass |
| "CPU only: memory-bound" | host has an RTX 5090 running other tenants' work, and the split table shows `nthreads` changing timings | reworded: implementation is CPU-only; it is memory-bound on a quiet host and was measured on a busy one |
| usage example | ran it; `np.array_equal` True | reproduced |

The `bench/analysis.md` table's `decompress2` figures are up to about 2x lower
than today's direct measurements of the same call, and the phase-to-phase noise
on this host reaches that magnitude. That file's provenance is not from this
session, so its row-level numbers should not be treated as verified beyond the
one row checked above.

## Unchecked work: was work skipped on one side?

The concern was a row where one side does less. Three things were checked:

1. Params and inputs are identical between sides (see above).
2. Results are verified: every compress row cross-decompresses the other side's
   output back to the original bytes, every decompress row compares both
   sides' output with the original, and from this pass every timed call's
   output is hashed and compared between the two sides on decompress rows.
   That check was added after its absence let a bad row print `1.20x` for
   `decompress memcpy chunk` (upstream 1.13 ms against a normally 0.72 ms
   reading) with nothing flagging it.
3. The row that moved most, `compress random bytes, 8 MiB`, was investigated
   because its relative had moved from 0.87x to 0.79x. Mojo's `compress` does
   skip the shuffle when `typesize == 1`, but that is the identity transform,
   so the skipped work produces no byte of output. The benchmark avoids the
   question entirely: it passes `typesize=1, filters=[NOFILTER...]`, so neither
   side runs a filter. Upstream's own `typesize=1` path was checked to
   round-trip to the same bytes. Conclusion: no work is skipped on one side in
   this row or, as far as the harness can show, in any other.

## Single-benchmark variation

One `pixi run bench` reading per row is not trustworthy on this host. The same
row was seen at, for example, `compress int32` 1.77 to 2.15 ms, `compress random
bytes` 2.68 to 3.75 ms, and `decompress memcpy` 0.74 to 1.12 ms. The README
therefore reports medians of many runs and lists the individual readings here.
Twenty percent of the relative column is comfortably inside the host's noise on
the decompression rows.

## Tests: parity assertions

`git log -p -- tests/` shows three commits touching tests (`fe2e08c`, `b86a63f`,
`54a4b91`). Only one line was ever removed, `b86a63f`'s replacement of
`parametrize("size", [32 MiB - 4, 32 MiB])` with
`[32 MiB - 4, 32 MiB, 32 MiB + 3]`; that is a strengthened test, not a loosened
one. No assertion, tolerance, `skip`, or `xfail` was removed or relaxed, and
there is no float tolerance anywhere (the tests compare bytes). All parity
tests still import upstream `blosc2` and assert against it:
`test_mojo_chunks_decode_upstream`, `test_upstream_chunks_decode_mojo`,
`test_mojo_special_zero_chunk_decodes_upstream`,
`test_incompressible_input_uses_standard_memcpy_chunk`,
`test_repeated_compression_does_not_reuse_stale_hash_entries`,
`test_compress2_cross_compatibility_with_explicit_pipeline`,
`test_decompress2_accepts_upstream_never_split_chunk`,
`test_pack_array_cross_compatibility`, `test_pack_and_unpack_cross_compatibility`,
`test_unsupported_upstream_codec_is_reported`, and
`test_public_signatures_match_upstream`. Nothing was restored, because nothing
was found to restore.

Added: `test_byte_shuffle_below_split_threshold_is_complete`, which pins the
16 MiB sub-threshold shuffle/unshuffle path to the reference byte layout for
`typesize` 4 and 8 at 2 and 4 threads. It passes. It closes a coverage gap, not
a defect.

## A defect I claimed and then disproved

While reading `shuffle_elements`/`unshuffle_elements` I concluded that the split
could write outside a worker's range (because the scalar loop runs `while
element < stop` but indexes a plane that starts at the buffer's element 0), and
I wrote a section of the README saying so. It is false. `byte_shuffle` and
`byte_unshuffle` at 32, 33, 40, 48 and 64 MiB, `nthreads` 1/2/4/8, match each
other and a reference layout byte for byte, with zero differing bytes, and
round-trip to the source. The README section now records the check instead of
the claim. This is the reason the final verification ran everything again after
the wording changes.

## A reproduction that never existed

An earlier version of this file cited a 16 MiB shuffle "costing twice the bytes
it moves" (3.9 ms against 8.1 ms for 32 MiB). That was a coincidence of two
timing protocols in one process: a best-of-7 32 MiB reading taken immediately
after 16 MiB runs that had just evicted the warm buffers. Run plainly and
repeatedly, 16 MiB takes 2.74-3.0 ms and 32 MiB 26-28 ms. There is no defect
and no test was kept for it.

## What is not verified

- The host cannot give clean absolute numbers. Every figure in the README is a
  median of this session's readings on this host; reproducing it on a quiet
  machine will give different absolute times, and the relative column is the
  part that should carry over.
- `bench/analysis.md`'s full table (277 rows of hotspots) was not re-run. Only
  the one row behind the README's 0.51 gate figure was checked.
- GPU and CPU-parallel paths were not attempted; `bench/analysis.md` and the
  README's Parallelism section state why the toolchain does not expose them.
