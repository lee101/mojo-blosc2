"""Benchmark the covered codec path against Python-Blosc2."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
import time

import blosc2 as upstream
import numpy as np

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojo_blosc2 as mojo  # noqa: E402


def best_time(function, repetitions=7):
    function()
    best = float("inf")
    result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        result = function()
        elapsed = time.perf_counter() - start
        digest = output_digest(result)
        best = min(best, elapsed)
    return best, result, digest


def output_digest(result):
    if isinstance(result, (bytes, bytearray)):
        return hashlib.sha256(result).hexdigest()
    memory = memoryview(result)
    try:
        # `memory.cast("B")` keeps a reference to the cast view, so the only
        # handle that may be released twice here is the original `memory`.
        view = memory.cast("B") if memory.format not in ("B", "b") else memory
        return hashlib.sha256(view).hexdigest()
    finally:
        memory.release()


def machine():
    model = "unknown CPU"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
            for line in cpuinfo:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return (
        f"{model}; {platform.system()} {platform.machine()}; "
        f"Python {platform.python_version()}"
    )


def main():
    count = 2_000_000
    integers = np.arange(count, dtype=np.int32)
    x = np.linspace(0, 400, count, dtype=np.float64)
    smooth = np.sin(x) + x * 0.001
    random_bytes = np.random.default_rng(7).integers(
        0, 256, size=8 * 1024 * 1024, dtype=np.uint8
    )

    def mojo_params(typesize, filter_code):
        return {
            "codec": mojo.Codec.LZ4,
            "typesize": typesize,
            "clevel": 5,
            "blocksize": 256 * 1024,
            "nthreads": 1,
            "splitmode": mojo.SplitMode.AUTO_SPLIT,
            "filters": [mojo.Filter.NOFILTER] * 5 + [filter_code],
            "filters_meta": [0] * 6,
        }

    def upstream_params(typesize, filter_code):
        return {
            "codec": upstream.Codec.LZ4,
            "typesize": typesize,
            "clevel": 5,
            "blocksize": 256 * 1024,
            "nthreads": 1,
            "splitmode": upstream.SplitMode.AUTO_SPLIT,
            "filters": [upstream.Filter.NOFILTER] * 5 + [filter_code],
            "filters_meta": [0] * 6,
        }

    encoded_integer = upstream.compress2(
        integers, **upstream_params(4, upstream.Filter.SHUFFLE)
    )
    encoded_smooth = upstream.compress2(
        smooth, **upstream_params(8, upstream.Filter.SHUFFLE)
    )
    encoded_random = upstream.compress2(
        random_bytes, **upstream_params(1, upstream.Filter.NOFILTER)
    )

    if os.environ.get("MOJO_BENCH_PROFILE"):
        shuffled_integers = mojo.byte_shuffle(integers, 4)
        shuffled_smooth = mojo.byte_shuffle(smooth, 8)
        encoded_integer_nofilter = upstream.compress2(
            integers, **upstream_params(4, upstream.Filter.NOFILTER)
        )
        large_block_params = mojo_params(8, mojo.Filter.SHUFFLE)
        large_block_params["blocksize"] = 1024 * 1024
        encoded_smooth_large_blocks = upstream.compress2(
            smooth,
            **{
                **upstream_params(8, upstream.Filter.SHUFFLE),
                "blocksize": 1024 * 1024,
            },
        )
        large_integers = np.tile(integers, 8)
        shuffled_large_integers = mojo.byte_shuffle(
            large_integers, 4, nthreads=1
        )
        print(f"Machine: {machine()}")
        print()
        print("| diagnostic | time |")
        print("| --- | ---: |")
        for name, function in [
            (
                "shuffle int32, 8 MB",
                lambda: mojo.byte_shuffle(integers, 4),
            ),
            (
                "unshuffle int32, 8 MB",
                lambda: mojo.byte_unshuffle(shuffled_integers, 4),
            ),
            (
                "shuffle float64, 16 MB",
                lambda: mojo.byte_shuffle(smooth, 8),
            ),
            (
                "unshuffle float64, 16 MB",
                lambda: mojo.byte_unshuffle(shuffled_smooth, 8),
            ),
            (
                "compress int32 NOFILTER, 8 MB",
                lambda: mojo.compress2(
                    integers, **mojo_params(4, mojo.Filter.NOFILTER)
                ),
            ),
            (
                "decompress int32 NOFILTER, 8 MB",
                lambda: mojo.decompress2(encoded_integer_nofilter, nthreads=1),
            ),
            (
                "compress float64 1 MiB blocks, 1 thread",
                lambda: mojo.compress2(
                    smooth, **{**large_block_params, "nthreads": 1}
                ),
            ),
            (
                "compress float64 1 MiB blocks, 2 threads",
                lambda: mojo.compress2(
                    smooth, **{**large_block_params, "nthreads": 2}
                ),
            ),
            (
                "decompress float64 1 MiB blocks, 1 thread",
                lambda: mojo.decompress2(
                    encoded_smooth_large_blocks, nthreads=1
                ),
            ),
            (
                "decompress float64 1 MiB blocks, 2 threads",
                lambda: mojo.decompress2(
                    encoded_smooth_large_blocks, nthreads=2
                ),
            ),
            (
                "shuffle int32, 64 MB, 1 thread",
                lambda: mojo.byte_shuffle(
                    large_integers, 4, nthreads=1
                ),
            ),
            (
                "shuffle int32, 64 MB, 4 threads",
                lambda: mojo.byte_shuffle(
                    large_integers, 4, nthreads=4
                ),
            ),
            (
                "unshuffle int32, 64 MB, 1 thread",
                lambda: mojo.byte_unshuffle(
                    shuffled_large_integers, 4, nthreads=1
                ),
            ),
            (
                "unshuffle int32, 64 MB, 4 threads",
                lambda: mojo.byte_unshuffle(
                    shuffled_large_integers, 4, nthreads=4
                ),
            ),
        ]:
            seconds, _, _ = best_time(function)
            print(f"| {name} | {seconds * 1000:.2f} ms |")
        print()

    cases = [
        (
            "compress int32 arange, 8 MB",
            lambda: mojo.compress2(
                integers, **mojo_params(4, mojo.Filter.SHUFFLE)
            ),
            lambda: upstream.compress2(
                integers, **upstream_params(4, upstream.Filter.SHUFFLE)
            ),
            "compress",
            integers.tobytes(),
        ),
        (
            "decompress int32 arange, 8 MB",
            lambda: mojo.decompress2(encoded_integer, nthreads=1),
            lambda: upstream.decompress2(encoded_integer, nthreads=1),
            "decompress",
            integers.tobytes(),
        ),
        (
            "compress smooth float64, 16 MB",
            lambda: mojo.compress2(
                smooth, **mojo_params(8, mojo.Filter.SHUFFLE)
            ),
            lambda: upstream.compress2(
                smooth, **upstream_params(8, upstream.Filter.SHUFFLE)
            ),
            "compress",
            smooth.tobytes(),
        ),
        (
            "decompress smooth float64, 16 MB",
            lambda: mojo.decompress2(encoded_smooth, nthreads=1),
            lambda: upstream.decompress2(encoded_smooth, nthreads=1),
            "decompress",
            smooth.tobytes(),
        ),
        (
            "compress random bytes, 8 MiB",
            lambda: mojo.compress2(
                random_bytes, **mojo_params(1, mojo.Filter.NOFILTER)
            ),
            lambda: upstream.compress2(
                random_bytes,
                **upstream_params(1, upstream.Filter.NOFILTER),
            ),
            "compress_native_typesize_one",
            random_bytes.tobytes(),
        ),
        (
            "decompress memcpy chunk, 8 MiB",
            lambda: mojo.decompress2(encoded_random, nthreads=1),
            lambda: upstream.decompress2(encoded_random, nthreads=1),
            "decompress",
            random_bytes.tobytes(),
        ),
    ]

    print(f"Machine: {machine()}")
    mojo_version = subprocess.run(
        ["mojo", "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    print(f"Mojo: {mojo_version}")
    print(f"Upstream: blosc2 {upstream.__version__}; nthreads=1")
    print()
    print("| case | mojo-blosc2 | upstream blosc2 | relative |")
    print("| --- | ---: | ---: | ---: |")
    for name, mojo_function, upstream_function, operation, original in cases:
        mojo_seconds, mojo_result, mojo_digest = best_time(mojo_function)
        upstream_seconds, upstream_result, upstream_digest = best_time(
            upstream_function
        )
        if operation.startswith("compress"):
            if operation == "compress_native_typesize_one":
                # typesize=1 byte shuffle is the identity, so upstream's own
                # native typesize=1 call must round-trip to the original bytes
                # as well; this keeps that assumption pinned to upstream.
                native = upstream.compress(
                    random_bytes, codec=upstream.Codec.LZ4, clevel=5
                )
                if upstream.decompress2(native) != original:
                    raise AssertionError(
                        f"upstream native typesize=1 path changed for {name}"
                    )
            if upstream.decompress2(mojo_result) != original:
                raise AssertionError(f"Mojo result failed verification for {name}")
            if mojo.decompress2(upstream_result) != original:
                raise AssertionError(
                    "upstream result failed verification for "
                    f"{name}: {len(upstream_result)} bytes, upstream round-trips "
                    f"to {upstream.decompress2(upstream_result) == original}"
                )
        elif mojo_result != upstream_result or mojo_result != original:
            raise AssertionError(f"benchmark outputs differ for {name}")
        # The only cross-side digest that must match is between the two sides
        # of a decompression row: both decode the exact same upstream chunk.
        # Compression output is permitted to differ in bytes; it is checked
        # above by round-tripping through the other implementation instead.
        if operation == "decompress" and mojo_digest != upstream_digest:
            raise AssertionError(
                f"decompressed bytes differ between sides for {name}"
            )
        relative = upstream_seconds / mojo_seconds
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{upstream_seconds * 1000:.2f} ms | {relative:.2f}x |"
        )


if __name__ == "__main__":
    main()
