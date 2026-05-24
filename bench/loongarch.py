#!/usr/bin/env python3
"""
Microbenchmark suite for the LoongArch / LSX / LASX / Highway optimizations
on the loongson-experimental branch.

For each op we touched, we compare two timings on the SAME build:

  CONTIG  — contiguous input arrays. The dispatcher's step check passes,
            so the NPYV / Highway path runs.
  STRIDED — same data laid out in a 2N-element array sliced [::2]. The
            dispatcher's step check fails, falling back to scalar libm
            inside an UNARY_LOOP / BINARY_LOOP.

Same compiled binary on both sides, so the ratio isolates the NPYV/Highway
win from everything else (compiler version, libc, NUMA, etc.).

Usage on the 3A6000 target:

    spin python -- bench/loongarch.py
    spin python -- bench/loongarch.py --quick
    spin python -- bench/loongarch.py --n 5000000 --iters 30

Run-to-run noise is around ±15% for fast ops; for stable numbers re-run
with larger --iters.

Output is plain text to stdout. To save, just redirect:

    spin python -- bench/loongarch.py > .benchresults/$(date -I)-loongarch.txt

This script lives in-tree at bench/loongarch.py — see OPTIMIZATIONS.md and
BENCHMARKS.md for the curated tables and the algorithmic background.
"""
from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass

import numpy as np


# Lane ranges chosen so the polynomial path is the typical case for that
# op (e.g. we want |x| in arctanh's domain (-1, 1), positive x for log,
# etc.). The exact range doesn't change "is the NPYV path running" — it
# just keeps results meaningful.
@dataclass
class UnarySpec:
    attr: str                    # numpy attribute name (e.g. "log1p")
    lo: float
    hi: float
    # Group label printed before the table row.
    group: str


@dataclass
class BinarySpec:
    attr: str
    lo1: float
    hi1: float
    lo2: float
    hi2: float
    group: str


# Unary ops, grouped by which dispatch file owns them.
UNARY: list[UnarySpec] = [
    # loops_exponent_log
    UnarySpec("exp",    -2.0,  2.0, "loops_exponent_log"),
    UnarySpec("log",     0.5,  4.0, "loops_exponent_log"),
    # loops_trigonometric (Highway-backed sin/cos; tan went through
    # loops_umath_fp before our Tier-2)
    UnarySpec("sin",    -3.0,  3.0, "loops_trigonometric (Highway)"),
    UnarySpec("cos",    -3.0,  3.0, "loops_trigonometric (Highway)"),
    # loops_hyperbolic (Highway-backed tanh)
    UnarySpec("tanh",   -3.0,  3.0, "loops_hyperbolic (Highway)"),
    # loops_umath_fp — Tier 1 (compose on exp/log)
    UnarySpec("sinh",   -2.0,  2.0, "loops_umath_fp Tier 1"),
    UnarySpec("cosh",   -2.0,  2.0, "loops_umath_fp Tier 1"),
    UnarySpec("exp2",   -3.0,  3.0, "loops_umath_fp Tier 1"),
    UnarySpec("log2",    0.5,  4.0, "loops_umath_fp Tier 1"),
    UnarySpec("log10",   0.5,  4.0, "loops_umath_fp Tier 1"),
    UnarySpec("arccosh", 1.5,  4.0, "loops_umath_fp Tier 1"),
    # loops_umath_fp — Tier 2
    UnarySpec("log1p",  -0.5,  0.5, "loops_umath_fp Tier 2"),
    UnarySpec("expm1",  -0.5,  0.5, "loops_umath_fp Tier 2"),
    UnarySpec("arcsinh",-2.0,  2.0, "loops_umath_fp Tier 2"),
    UnarySpec("arctanh",-0.5,  0.5, "loops_umath_fp Tier 2"),
    UnarySpec("arctan", -3.0,  3.0, "loops_umath_fp Tier 2"),
    UnarySpec("arcsin", -0.9,  0.9, "loops_umath_fp Tier 2"),
    UnarySpec("arccos", -0.9,  0.9, "loops_umath_fp Tier 2"),
    UnarySpec("tan",    -1.4,  1.4, "loops_umath_fp Tier 2"),
    UnarySpec("cbrt",   -3.0,  3.0, "loops_umath_fp Tier 2"),
]

# Binary ops.
BINARY: list[BinarySpec] = [
    # loops_arithm_fp — the bread-and-butter element-wise float arithmetic.
    # Divisor range kept away from 0. These are where the LASX-vs-LSX width
    # question lives: add/sub/mul are largely memory-bound, divide is
    # throughput-bound on the FP divider (the more likely LASX winner).
    BinarySpec("add",      -1.0e3, 1.0e3, -1.0e3, 1.0e3, "loops_arithm_fp"),
    BinarySpec("subtract", -1.0e3, 1.0e3, -1.0e3, 1.0e3, "loops_arithm_fp"),
    BinarySpec("multiply", -1.0e3, 1.0e3, -1.0e3, 1.0e3, "loops_arithm_fp"),
    BinarySpec("divide",    0.5,   4.0,   0.5,    4.0,   "loops_arithm_fp"),
    BinarySpec("arctan2", -1.0, 1.0, 0.5, 2.5, "loops_umath_fp Tier 2"),
    BinarySpec("power",    0.5, 4.0, 0.5, 4.0, "loops_umath_fp Tier 2"),
]


def time_call(fn, *arrays, iters: int) -> float:
    """Return mean per-call time in seconds."""
    for _ in range(3):
        fn(*arrays)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*arrays)
    return (time.perf_counter() - t0) / iters


def melem_per_sec(n: int, dt: float) -> float:
    return n / dt / 1e6


def fmt_row(op: str, dtype_name: str, contig: float, strided: float | None) -> str:
    dt_short = {
        "float32": "f32",
        "float64": "f64",
        "int16":   "i16",
        "int32":   "i32",
        "int64":   "i64",
        "bool":    "bool",
    }.get(dtype_name, dtype_name)
    if strided is None or strided <= 0:
        return f"  {op:<13}{dt_short:<6}{contig:>10.1f}{'':>13}{'':>10}"
    speedup = contig / strided
    return (f"  {op:<13}{dt_short:<6}"
            f"{contig:>10.1f}{strided:>13.1f}{speedup:>9.2f}x")


def run_unary(n: int, iters: int, dtypes: tuple[type, ...]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for spec in UNARY:
        fn = getattr(np, spec.attr)
        for dtype in dtypes:
            cont = np.linspace(spec.lo, spec.hi, n).astype(dtype)
            wide = np.linspace(spec.lo, spec.hi, 2 * n).astype(dtype)
            strided = wide[::2]
            c_t = time_call(fn, cont, iters=iters)
            s_t = time_call(fn, strided, iters=iters)
            row = fmt_row(spec.attr, dtype.__name__,
                          melem_per_sec(n, c_t),
                          melem_per_sec(n, s_t))
            out.setdefault(spec.group, []).append(row)
    return out


def run_binary(n: int, iters: int, dtypes: tuple[type, ...]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for spec in BINARY:
        fn = getattr(np, spec.attr)
        for dtype in dtypes:
            a = np.linspace(spec.lo1, spec.hi1, n).astype(dtype)
            b = np.linspace(spec.lo2, spec.hi2, n).astype(dtype)
            aw = np.linspace(spec.lo1, spec.hi1, 2 * n).astype(dtype)
            bw = np.linspace(spec.lo2, spec.hi2, 2 * n).astype(dtype)
            astrided, bstrided = aw[::2], bw[::2]
            c_t = time_call(fn, a, b, iters=iters)
            s_t = time_call(fn, astrided, bstrided, iters=iters)
            row = fmt_row(spec.attr, dtype.__name__,
                          melem_per_sec(n, c_t),
                          melem_per_sec(n, s_t))
            out.setdefault(spec.group, []).append(row)
    return out


def run_logical(n: int, iters: int) -> list[str]:
    rng = np.random.default_rng(0)
    out = []
    b = (rng.random(n) > 0.5)
    bw = (rng.random(2 * n) > 0.5)
    bs = bw[::2]
    c_t = time_call(np.logical_not, b, iters=iters)
    s_t = time_call(np.logical_not, bs, iters=iters)
    out.append(fmt_row("logical_not", "bool",
                       melem_per_sec(n, c_t),
                       melem_per_sec(n, s_t)))
    c_t = time_call(np.logical_and, b, b, iters=iters)
    s_t = time_call(np.logical_and, bs, bs, iters=iters)
    out.append(fmt_row("logical_and", "bool",
                       melem_per_sec(n, c_t),
                       melem_per_sec(n, s_t)))
    return out


def run_f16_unary(n: int, iters: int) -> list[str]:
    """f16 transcendentals via the LASX f16<->f32 bridge.

    Same op list as UNARY, restricted to ops that have an f32 NPYV kernel
    (which is all of them after the bridge work). Bench compares the
    contiguous bridged path against the strided scalar libm fallback.
    """
    out = []
    for spec in UNARY:
        fn = getattr(np, spec.attr)
        cont = np.linspace(spec.lo, spec.hi, n).astype(np.float16)
        wide = np.linspace(spec.lo, spec.hi, 2 * n).astype(np.float16)
        strided = wide[::2]
        c_t = time_call(fn, cont, iters=iters)
        s_t = time_call(fn, strided, iters=iters)
        out.append(fmt_row(spec.attr, "float16",
                           melem_per_sec(n, c_t),
                           melem_per_sec(n, s_t)))
    return out


def run_integer(n: int, iters: int) -> list[str]:
    """Integer arithmetic: add / multiply / floor_divide / mod / less /
    max-reduce across int{8,16,32,64} and uint{8,16,32,64}.

    Most of these dispatch through loops_arithmetic, loops_modulo,
    loops_comparison, and loops_minmax — all of which now have LSX or
    LASX targets on LoongArch.

    "Strided" here is the slice-by-2 of a double-length array; for ops
    that don't have a strided SIMD fallback this gives us the scalar
    reference speed.
    """
    out = []
    rng = np.random.default_rng(0)
    int_dtypes = (np.int8, np.int16, np.int32, np.int64,
                  np.uint8, np.uint16, np.uint32, np.uint64)
    for dt in int_dtypes:
        info = np.iinfo(dt)
        if np.issubdtype(dt, np.signedinteger):
            a = rng.integers(info.min // 2, info.max // 2, n).astype(dt)
        else:
            a = rng.integers(0, min(info.max, 1 << 30), n).astype(dt)
        b = rng.integers(1, 100, n).astype(dt)
        aw = rng.integers(info.min // 2 if np.issubdtype(dt, np.signedinteger) else 0,
                          info.max // 2 if np.issubdtype(dt, np.signedinteger) else min(info.max, 1 << 30),
                          2 * n).astype(dt)
        bw = rng.integers(1, 100, 2 * n).astype(dt)
        astrided, bstrided = aw[::2], bw[::2]
        for op_name in ("add", "multiply", "floor_divide", "mod", "less"):
            fn = getattr(np, op_name)
            c_t = time_call(fn, a, b, iters=iters)
            s_t = time_call(fn, astrided, bstrided, iters=iters)
            out.append(fmt_row(op_name, dt.__name__,
                               melem_per_sec(n, c_t),
                               melem_per_sec(n, s_t)))
        # Reductions: just the contiguous time, no strided comparison
        for op_name in ("max", "argmax"):
            fn = getattr(np, op_name)
            c_t = time_call(fn, a, iters=iters)
            out.append(fmt_row(op_name, dt.__name__,
                               melem_per_sec(n, c_t), None))
    return out


def run_sort(n: int, iters: int) -> list[str]:
    """np.sort has no strided dispatch comparison — just report contig timing."""
    rng = np.random.default_rng(0)
    out = []
    for dtype in (np.float32, np.float64, np.int16, np.int32, np.int64):
        if np.issubdtype(dtype, np.integer):
            arr = rng.integers(0, 1 << 14, n).astype(dtype)
        else:
            arr = rng.random(n).astype(dtype)
        # np.sort sorts in-place when called as a method, but np.sort returns
        # a sorted copy — so we don't need to refresh between iterations.
        # Sort cost varies with input order; bench against the same input.
        sort_iters = max(2, iters // 3)  # sort is heavier than ufuncs
        t = time_call(lambda a: np.sort(a), arr, iters=sort_iters)
        out.append(fmt_row("sort", dtype.__name__,
                           melem_per_sec(n, t), None))
    return out


def print_section(title: str, rows: list[str]) -> None:
    if not rows:
        return
    print(f"\n--- {title} ---")
    print(f"  {'op':<13}{'dtype':<6}{'NPYV (M/s)':>10}{'scalar (M/s)':>13}{'speedup':>10}")
    for r in rows:
        print(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1_000_000,
                    help="Array length (default: 1,000,000)")
    ap.add_argument("--iters", type=int, default=15,
                    help="Iterations per measurement (default: 15)")
    ap.add_argument("--quick", action="store_true",
                    help="Smaller n / fewer iters for fast feedback")
    ap.add_argument("--skip-sort", action="store_true",
                    help="Skip sort benchmarks (they're slow)")
    args = ap.parse_args()

    if args.quick:
        n, iters = 100_000, 5
    else:
        n, iters = args.n, args.iters

    # Header — captures enough info that the output is self-documenting.
    print("=" * 72)
    print(f"  loongarch.py — n={n:,}  iters={iters}")
    print(f"  numpy  : {np.__version__}")
    print(f"  python : {platform.python_version()}")
    print(f"  uname  : {platform.system()} {platform.release()} {platform.machine()}")
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name") or line.startswith("CPU MHz"):
                    print(f"  cpu    : {line.split(':', 1)[1].strip()}")
                    break
    except (OSError, IndexError):
        pass
    # SIMD info — read from numpy's build-time config dict directly so we
    # don't get show_runtime's pprint noise.
    try:
        ext = np.__config__.CONFIG["SIMD Extensions"]  # type: ignore[attr-defined]
        print(f"  simd   : baseline={ext.get('baseline', [])} "
              f"dispatched={ext.get('found', [])}")
    except (AttributeError, KeyError, TypeError):
        pass
    print("=" * 72)

    dtypes = (np.float32, np.float64)

    print("\nUnary ops — contiguous NPYV vs strided scalar libm")
    unary_results = run_unary(n, iters, dtypes)
    # Print in a stable order grouped by dispatch file
    for group in ("loops_exponent_log",
                  "loops_trigonometric (Highway)",
                  "loops_hyperbolic (Highway)",
                  "loops_umath_fp Tier 1",
                  "loops_umath_fp Tier 2"):
        print_section(group, unary_results.get(group, []))

    print("\nBinary ops — contiguous NPYV vs strided scalar libm")
    binary_results = run_binary(n, iters, dtypes)
    for group in ("loops_arithm_fp", "loops_umath_fp Tier 2"):
        print_section(group, binary_results.get(group, []))

    print("\nLogical ops — Highway")
    print_section("loops_logical (Highway)", run_logical(n, iters))

    # f16: same unary spec list, but dtype=float16 — exercises the
    # LASX f16↔f32 bridge in loops_half.dispatch.c.src.
    print("\nf16 unary ops — LASX f16<->f32 bridge")
    print_section("loops_half (f16 bridge)",
                  run_f16_unary(n, iters))

    print("\nInteger arithmetic — contiguous NPYV vs strided scalar libm")
    print_section("integer ops", run_integer(n, iters))

    if not args.skip_sort:
        print("\nSort — Highway vqsort via LSX (and LASX where dispatched)")
        print_section("highway_qsort", run_sort(n, iters))

    return 0


if __name__ == "__main__":
    sys.exit(main())
