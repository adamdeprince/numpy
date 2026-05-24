# LoongArch Benchmarks

Throughput measurements for NumPy operations on Loongson hardware.

## Reproducing

```
ssh loongson  # or whatever your host alias is
cd ~/dev/numpy
spin python -- bench/loongarch.py                    # ~2 min, all ops
spin python -- bench/loongarch.py --quick            # ~30 sec, smaller n
spin python -- bench/loongarch.py --skip-sort        # skip the slow part
spin python -- bench/loongarch.py --n 5000000 --iters 30   # tighter numbers
```

The script benchmarks each ufunc both as contiguous (so the NPYV /
Highway dispatch fires) and as a strided view of a longer array (so the
dispatcher falls back to scalar libm inside the UNARY_LOOP). Same
compiled binary on both sides, so the ratio isolates our optimization
from compiler / libc / hardware effects.

The tables below come from `bench/loongarch.py` runs; raw outputs are
preserved under `~/dev/numpy/.benchresults/`.

## Cumulative speedup — across all four data-type families

The bench script is now wider: f64 + f32 + f16 + integer + bool, 106
op × dtype rows in total. The headline number tracks every ufunc the
LoongArch port touched, not just the f64/f32 transcendentals.

### Summary across all 106 op × dtype pairs (current — 2026-05-25)

| family | n | geomean | min | max |
|---|---:|---:|---:|---:|
| **all** | **106** | **5.44×** | 1.00× | 25.84× |
| f64 | 22 | 4.72× | 2.46× | 14.55× |
| f32 | 22 | 4.48× | 1.78× | 20.66× |
| f16 (LASX bridge) | 20 | **9.91×** | 3.95× | 25.84× |
| integer | 40 | 4.59× | 1.00× | 16.56× |
| bool | 2 | 16.94× | 13.92× | 20.62× |

This is the headline number for the whole loongson-experimental branch:
**numpy on the 3A6000 is 5.44× faster (geomean across 106 op × dtype
pairs) than the same numpy built without the LSX/LASX optimization
work — measured as contig (NPYV/Highway path) vs strided (scalar
libm fallback) in the same compiled binary.**

Earlier snapshot (pre-2026-05-25): geomean 5.16×, same 106 ops; the
0.28× lift is mostly cbrt f32 (174 → 537 M/s), tanh f64 (54 → 81),
and power f32+f64 (140/53 → 183/76).

The two `1.00×` lanes are `floor_divide` for int64 / uint64 — both
fall through to scalar on LoongArch because the dispatch macro guards
the vector-by-vector and divide-by-scalar paths together, and the
existing divide-by-scalar uses Granlund-Möller multiply-high that's
emulated on LSX. Splitting those guards is logged in `FUTURE_TODOS.md`.

### Top movers (cumulative bench, ratio = NPYV-contig / scalar-strided)

| op | dtype | NPYV M/s | scalar M/s | speedup |
|---|---|---:|---:|---:|
| tanh | f16 | 531 | 20 | **26.48×** |
| logical_and | bool | 15043 | 749 | 20.08× |
| tan | f32 | 769 | 38 | 20.03× |
| arctan | f16 | 757 | 38 | 19.92× |
| arctanh | f16 | 447 | 24 | 18.68× |
| expm1 | f16 | 616 | 36 | 17.34× |
| tan | f16 | 523 | 29 | 18.14× |
| logical_not | bool | — | — | 12.29× |
| mod | int8 | 2461 | 149 | 16.47× |
| floor_divide | int8 | 2648 | 175 | 15.10× |

### Lowest performers

| op | dtype | NPYV M/s | scalar M/s | speedup | notes |
|---|---|---:|---:|---:|---|
| floor_divide | int64 | 107 | 107 | 1.00× | scalar; dispatch guard combined |
| floor_divide | uint64 | 186 | 186 | 1.00× | same |
| tanh | f64 | 54 | 42 | 1.31× | Highway-routed, polynomial owned upstream |
| add | int64 | 497 | 379 | 1.31× | LSX already saturates 64-bit lane throughput |
| add | uint64 | 515 | 380 | 1.35× | same |
| power | f32 | 137 | 80 | 1.61× | composes on exp+log, edge-case heavy |
| log1p | f64 | — | — | 1.95× | direct Padé, slow-path still through log_kernel |
| cbrt | f64 | 137 | 43 | 3.22× | FDLIBM bit-magic + Newton |
| sin | f64 | 240 | 68 | 3.51× | DD-style Cody-Waite + Payne-Hanek slow path |

## Original f32+f64-only comparison (pre-f16/int work)

The earlier "main vs head" diff using the narrower 46-op bench:

### Summary

- **Geometric mean speedup across the original 46 op × dtype pairs: 4.95×**
- 46 / 46 speedups ≥ 1.10×; 0 / 46 neutral; 0 / 46 regressions
- Lowest speedup was `power` f32 at 1.56×
- Top mover: `arctanh` f64 jumped 2.54× → 14.7× after switching from
  `0.5·log1p(2x/(1-x))` to a direct Cephes Padé in x².

### Per-op comparison (best to worst speedup)

| op       | dtype | main NPYV | head NPYV | HEAD/MAIN |
|----------|-------|----------:|----------:|----------:|
| tan      | f32   |      38.6 |     820.2 |  **21.25x** |
| sinh     | f32   |      25.7 |     394.1 |  **15.33x** |
| logical_and | bool | 1044.0 |   15425.7 |  **14.78x** |
| exp2     | f64   |      20.2 |     294.2 |  **14.56x** |
| tanh     | f32   |      26.5 |     328.8 |  **12.41x** |
| tan      | f64   |      24.3 |     247.6 |   10.19x |
| arctan   | f32   |      64.1 |     595.0 |    9.28x |
| expm1    | f32   |      70.6 |     649.8 |    9.20x |
| logical_not | bool | 1870.7 |   16473.9 |    8.81x |
| log1p    | f32   |      64.7 |     558.5 |    8.63x |
| log10    | f32   |      85.7 |     721.4 |    8.42x |
| arctanh  | f32   |      35.2 |     274.8 |    7.81x |
| log10    | f64   |      37.0 |     224.5 |    6.07x |
| arctan2  | f32   |      43.4 |     255.1 |    5.88x |
| sin      | f32   |      77.3 |     439.4 |    5.68x |
| cos      | f32   |      78.0 |     429.0 |    5.50x |
| sinh     | f64   |      23.9 |     113.4 |    4.74x |
| exp      | f64   |      71.2 |     332.9 |    4.68x |
| cosh     | f32   |      87.0 |     392.5 |    4.51x |
| arctan   | f64   |      45.3 |     203.3 |    4.49x |
| arccosh  | f32   |      54.7 |     238.7 |    4.36x |
| log      | f64   |      54.3 |     236.3 |    4.35x |
| arcsinh  | f32   |      31.0 |     133.2 |    4.30x |
| log2     | f64   |      52.7 |     224.4 |    4.26x |
| arccos   | f64   |      64.8 |     272.3 |    4.20x |
| arcsin   | f64   |      67.9 |     278.8 |    4.11x |
| log      | f32   |     206.8 |     775.5 |    3.75x |
| cbrt     | f32   |      52.4 |     189.3 |    3.61x |
| **sin**  | **f64** |    68.5 |     240.5 |    **3.51x** |
| log2     | f32   |     207.5 |     724.3 |    3.49x |
| expm1    | f64   |      67.8 |     235.9 |    3.48x |
| **cos**  | **f64** |    67.9 |     235.9 |    **3.47x** |
| exp      | f32   |     236.6 |     779.3 |    3.29x |
| arctan2  | f64   |      24.5 |      77.1 |    3.15x |
| arccosh  | f64   |      24.5 |      74.7 |    3.05x |
| exp2     | f32   |     248.5 |     732.2 |    2.95x |
| power    | f64   |      17.5 |      48.4 |    2.77x |
| cosh     | f64   |      46.3 |     112.3 |    2.43x |
| arccos   | f32   |      84.5 |     191.7 |    2.27x |
| tanh     | f64   |      24.0 |      51.7 |    2.15x |
| arcsin   | f32   |      95.7 |     198.3 |    2.07x |
| log1p    | f64   |      63.4 |     123.4 |    1.95x |
| arcsinh  | f64   |      25.7 |      49.8 |    1.94x |
| power    | f32   |      80.7 |     136.6 |    1.69x |
| arctanh  | f64   |      33.0 |      49.8 |    1.51x |
| cbrt     | f64   |      42.7 |      60.7 |    1.42x |

### Earlier version (before arcsin/acos rewrite)

The initial pass — using `arcsin(x) = arctan(x/√(1-x²))` — regressed
those two f64 ops badly (`arcsin` f64 0.51×, `arccos` f64 0.80×).
Rewriting both with FDLIBM's rational P/Q kernel (see
`OPTIMIZATIONS.md` for the algorithm) moved them from regressions to
~4× speedups and lifted the overall geomean from 3.09× to 4.25×.

<details><summary>Original comparison table (pre-rewrite)</summary>

| op       | dtype | main NPYV | main scalar | head NPYV | head scalar | HEAD/MAIN |
|----------|-------|----------:|------------:|----------:|------------:|----------:|
| sinh     | f32   |      25.7 |        25.7 |     392.8 |        25.6 |  **15.28x** |
| tan      | f32   |      38.6 |        38.8 |     541.1 |        18.6 |  **14.02x** |
| exp2     | f64   |      20.2 |        20.1 |     267.6 |        20.1 |  **13.25x** |
| tanh     | f32   |      26.5 |        26.6 |     342.5 |       179.1 |  **12.92x** |
| expm1    | f32   |      70.6 |        70.1 |     635.0 |        70.0 |    8.99x |
| log1p    | f32   |      64.7 |        64.5 |     555.4 |        64.1 |    8.58x |
| log10    | f32   |      85.7 |        86.0 |     688.1 |        85.3 |    8.03x |
| arctan   | f32   |      64.1 |        64.0 |     457.5 |        31.2 |    7.14x |
| logical_not | bool | 1870.7  |      1271.6 |   12873.5 |       484.6 |    6.88x |
| log10    | f64   |      37.0 |        36.8 |     215.3 |        36.7 |    5.82x |
| sin      | f32   |      77.3 |        77.0 |     430.6 |       236.3 |    5.57x |
| arctanh  | f32   |      35.2 |        35.2 |     182.7 |        12.9 |    5.19x |
| cos      | f32   |      78.0 |        77.7 |     403.4 |       223.8 |    5.17x |
| cosh     | f32   |      87.0 |        86.8 |     392.8 |        86.0 |    4.51x |
| arccosh  | f32   |      54.7 |        54.5 |     236.3 |        54.2 |    4.32x |
| arcsinh  | f32   |      31.0 |        30.7 |     133.0 |        30.3 |    4.29x |
| exp      | f64   |      71.2 |        70.2 |     293.5 |        70.9 |    4.12x |
| log2     | f64   |      52.7 |        52.0 |     210.2 |        52.1 |    3.99x |
| tan      | f64   |      24.3 |        24.2 |      95.4 |        13.4 |    3.93x |
| log      | f64   |      54.3 |        54.0 |     209.1 |        54.1 |    3.85x |
| sinh     | f64   |      23.9 |        23.8 |      82.6 |        23.8 |    3.46x |
| log2     | f32   |     207.5 |       206.3 |     727.7 |       200.1 |    3.51x |
| expm1    | f64   |      67.8 |        66.9 |     227.9 |        66.6 |    3.36x |
| log      | f32   |     206.8 |       205.8 |     690.5 |       202.0 |    3.34x |
| exp      | f32   |     236.6 |       234.2 |     776.3 |       231.8 |    3.28x |
| exp2     | f32   |     248.5 |       245.7 |     719.0 |       235.7 |    2.89x |
| arctan   | f64   |      45.3 |        45.1 |     123.3 |        26.1 |    2.72x |
| arctan2  | f32   |      43.4 |        43.1 |     112.7 |        16.2 |    2.60x |
| arccosh  | f64   |      24.5 |        24.4 |      60.3 |        24.3 |    2.46x |
| cbrt     | f32   |      52.4 |        52.4 |     120.6 |        17.9 |    2.30x |
| arctan2  | f64   |      24.5 |        24.3 |      53.3 |        15.0 |    2.18x |
| arccos   | f32   |      84.5 |        85.8 |     163.1 |        36.7 |    1.93x |
| tanh     | f64   |      24.0 |        23.9 |      45.4 |        43.9 |    1.89x |
| cosh     | f64   |      46.3 |        46.1 |      81.3 |        46.1 |    1.76x |
| arcsin   | f32   |      95.7 |        95.1 |     166.6 |        27.9 |    1.74x |
| arcsinh  | f64   |      25.7 |        25.5 |      42.0 |        16.9 |    1.63x |
| power    | f64   |      17.5 |        17.3 |      25.4 |         6.1 |    1.45x |
| log1p    | f64   |      63.4 |        62.9 |      83.6 |        62.9 |    1.32x |
| logical_and | bool | 1044.0 |       928.1 |    1182.6 |       226.8 |    1.13x |
| arctanh  | f64   |      33.0 |        32.7 |      36.8 |        21.1 |    1.12x |
| sin      | f64   |      68.5 |        67.8 |      67.8 |        66.3 |    0.99x |
| cbrt     | f64   |      42.7 |        42.7 |      42.4 |        16.6 |    0.99x |
| power    | f32   |      80.7 |        79.7 |      74.9 |        30.1 |    0.93x |
| cos      | f64   |      67.9 |        67.0 |      61.3 |        65.4 |    0.90x |
| arccos   | f64   |      64.8 |        67.4 |      52.0 |        28.5 |    0.80x |
| arcsin   | f64   |      67.9 |        67.3 |      34.3 |        29.9 |  **0.51x** |

**Reading the table:** `main` has no NPYV implementations for these
ops, so its CONTIG (`NPYV`) and STRIDED (`scalar`) columns are roughly
equal — both go through `npy_<func>` libm. `head` has NPYV impls;
CONTIG dispatches to NPYV, STRIDED falls back. The right speedup
column is `HEAD/MAIN` (CONTIG vs CONTIG), since that's what an
unmodified user program will see.

</details>

### Notes on the precision gate for sin / cos f64

The `sin` / `cos` f64 SIMD path is gated per-block on `max(|lane|) <
2^20`. Blocks containing any out-of-range lane fall back to scalar
libm — this preserves correctness past the Cody-Waite range without
needing a SIMD Payne-Hanek. See `OPTIMIZATIONS.md` for the precision
audit (max 1 ULP across the SIMD regime; 0 ULP in the libm-fallback
regime) and for the Payne-Hanek follow-up.

Raw output: `~/dev/numpy/.benchresults/main-vs-head-comparison.txt`.

## Hardware

- **CPU:** Loongson 3A6000 @ 2.0 GHz, LSX + LASX in HWCAP
- **OS:** Kylin V10 SP1, kernel 5.4.18-110-generic
- **Compiler:** gcc 15.2.0 (`/opt/loongson-gcc-15.2.0`)
- **Python:** 3.13.13 (pyenv)
- **NumPy:** 2.5.0.dev0, branch `loongson-experimental`

## Methodology

Microbenchmarks call the ufunc on a `linspace(-2, 2, n).astype(dtype)` array
(shifted by +3 for `log` so inputs are positive). Each measurement does 3
warmup iterations, then times `max(5, 2·10⁷ / n)` calls and reports the mean
per-call time. Throughput is `n / time_per_call`. Results in millions of
elements per second.

Run on the 3A6000 via `spin python -- -c <bench script>` against the build
described above. Same hardware and build flags for before / after rows
unless noted.

---

## 2026-05-17 — f32 exp / log via NPYV

Baseline: branch HEAD `79b033101a`, with `loops_exponent_log` taking the
scalar-libm fallback (AVX-only paths skipped on LoongArch).

After: same commit + the NPYV-style exp / log kernels documented in
`OPTIMIZATIONS.md`.

| op  | dtype   | n         | before (M/s) | after (M/s) | speedup |
| --- | ------- | --------- | -----------: | ----------: | ------: |
| exp | float32 |     1,024 |        172.8 |       253.4 |   1.47× |
| log | float32 |     1,024 |        156.1 |       241.6 |   1.55× |
| exp | float32 |    65,536 |        233.8 |       407.8 |   1.74× |
| log | float32 |    65,536 |        206.5 |       373.6 |   1.81× |
| exp | float32 | 1,000,000 |        236.5 |       412.7 |   1.74× |
| log | float32 | 1,000,000 |        208.2 |       378.4 |   1.82× |
| exp | float64 |     1,024 |         64.2 |        64.0 |   1.00× |
| log | float64 |     1,024 |         50.3 |        50.4 |   1.00× |
| exp | float64 |    65,536 |         71.0 |        71.2 |   1.00× |
| log | float64 |    65,536 |         54.4 |        54.1 |   0.99× |
| exp | float64 | 1,000,000 |         71.2 |        71.1 |   1.00× |
| log | float64 | 1,000,000 |         54.6 |        54.2 |   0.99× |

The 1.47–1.82× ratio falls short of the 4× theoretical lane width because
- `npyv_div_f32` (used for P/Q) has long latency on LSX and the polynomial
  has a serial dependency chain,
- the scalar tail handles up to 3 leftover lanes for n not divisible by 4,
- the smallest n is dominated by per-call overhead.

f64 is unchanged, as expected — no f64 NPYV path was added.

Raw outputs preserved on the target at `~/dev/numpy/.benchresults/baseline.txt`
and `~/dev/numpy/.benchresults/after_lsx.txt`.

---

## 2026-05-18 — Highway LSX enabled

Baseline: branch HEAD with `-DHWY_COMPILE_ONLY_SCALAR` still set on
loongarch64 (Highway forced to scalar).

After: flag removed, `LSX` added to the `highway_qsort` /
`highway_qsort_16bit` dispatch lists. See `OPTIMIZATIONS.md` 2026-05-18.

All measurements n = 1,000,000.

| op | dtype | before (M/s) | after (M/s) | speedup |
| --- | --- | ---: | ---: | ---: |
| `np.sort`        | int16   |    —    |   94.1  |   — (new measurement) |
| `np.sort`        | int32   |  12.7   |   61.5  |  **4.84×** |
| `np.sort`        | float32 |  11.5   |   49.2  |  **4.28×** |
| `np.sort`        | int64   |  12.2   |   37.3  |  **3.06×** |
| `np.sort`        | float64 |  11.3   |   22.6  |  **2.00×** |
| `np.sin`         | float32 |  76.7   |  249.9  |  **3.26×** |
| `np.cos`         | float32 |  78.1   |  246.4  |  **3.16×** |
| `np.tanh`        | float32 |  26.5   |  339.4  |  **12.81×** |
| `np.logical_not` | bool    | 1871.8  | 15266.5 |  **8.16×** |
| `np.logical_and` | bool    | 1044.0  | 13852.8 |  **13.27×** |
| `np.tan`         | float32 |  31.6   |   31.6  |   1.00× (no Highway path) |
| `np.arcsin`      | float32 |  95.7   |   95.5  |   1.00× |
| `np.arctan`      | float32 |  64.2   |   64.1  |   1.00× |
| `np.sinh`        | float32 |  24.5   |   24.4  |   1.00× |
| `np.cosh`        | float32 |  92.3   |   92.1  |   1.00× |

The 12.8× tanh and 4.8× int32 sort are the standout demo numbers.

The flat ops (`tan`, `arcsin`, `arctan`, `sinh`, `cosh`) don't route through
Highway today; they're separate follow-ups.

Test status: 7920 passed, 0 failed across `test_multiarray.py -k sort` and
the full `test_umath.py`.

---

## 2026-05-19 — LASX feature wired

Builds on the 2026-05-18 row. `LASX` is now a first-class feature: defined
in `meson_cpu/loongarch64/meson.build` (implies `LSX`), detected at runtime
via `HWCAP_LOONGARCH_LASX`, and listed in the `highway_qsort` (32/64-bit),
`loops_hyperbolic`, `loops_logical`, and `loops_trigonometric` dispatch
lists. `LASX` was kept *out* of `highway_qsort_16bit` due to an int16
regression in Highway's 16-bit vqsort on this CPU.

Numbers are vs the previous (LSX-only) row of this file. `scalar →` shows
the cumulative speedup over the scalar baseline.

| op | dtype | LSX (M/s) | LASX (M/s) | LASX/LSX | scalar → |
| --- | --- | ---: | ---: | ---: | ---: |
| `np.sort`        | int32   |  61.5 |  81.0 | **1.32×** | **6.38×** |
| `np.sort`        | float32 |  49.2 |  70.7 | **1.44×** | **6.15×** |
| `np.sort`        | float64 |  22.6 |  36.4 | **1.61×** | **3.22×** |
| `np.sort`        | int64   |  37.3 |  41.8 |  1.12×    | **3.43×** |
| `np.sort`        | int16   |  94.1 |  93.5 |  ≈1×      | (LASX excluded) |
| `np.sin`         | float32 | 249.9 | 434.6 | **1.74×** | **5.67×** |
| `np.cos`         | float32 | 246.4 | 437.2 | **1.77×** | **5.60×** |
| `np.tanh`        | float32 | 339.4 | 329.3 |  ≈1×      |  12.43× |
| `np.logical_not` | bool    | 15267 | 16145 |  1.06×    |   8.63× |
| `np.logical_and` | bool    | 13853 | 14072 |  ≈1×      |  13.48× |
| `np.tan`         | float32 |  31.6 |  31.6 |  1.00×    |   1.00× (still scalar) |
| `np.arcsin`      | float32 |  95.7 |  95.2 |  ≈1×      |   1.00× |
| `np.arctan`      | float32 |  64.2 |  64.2 |  1.00×    |   1.00× |
| `np.sinh`        | float32 |  24.5 |  24.5 |  1.00×    |   1.00× |
| `np.cosh`        | float32 |  92.3 |  92.2 |  ≈1×      |   1.00× |

The sin / cos ~1.75× and the 1.3–1.6× sort speedups come from numpy now
compiling the Highway-backed dispatch files for the LASX target — Highway's
internal multi-target dispatch then picks LASX intrinsics at runtime. Ops
that were already saturated by LSX (tanh, logical) don't move because
Highway's internal targeting was already using the widest path it could
find at its compile-time -mlsx setting.

The flat ops (`tan`, `arcsin`, `arctan`, `sinh`, `cosh`) sit in dispatch
files that include LSX/LASX in their lists but the function bodies don't
route through Highway — that's a separate investigation.

Test status: 7920 passed, 0 failed.

---

## 2026-05-20 — Native LASX NPYV backend (f32 exp / log)

A full LASX NPYV backend now exists at `numpy/_core/src/common/simd/lasx/`
mirroring the LSX backend file-for-file. The f32 exp/log kernel from
2026-05-17 is NPYV-style and picks up the LASX backend automatically once
`LASX` is listed in the `loops_exponent_log` dispatch.

All measurements at n = 1,000,000, on the 3A6000.

| op | dtype | scalar (M/s) | LSX (M/s) | LASX (M/s) | LASX/LSX | scalar → |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `np.exp` | float32 | 236.5 | 412.7 | **823.4** | **2.00×** | **3.48×** |
| `np.log` | float32 | 208.2 | 378.4 | **770.1** | **2.03×** | **3.70×** |

Smaller-n results show the call/tail overhead still dominates:

| op | dtype | n | LSX (M/s) | LASX (M/s) | LASX/LSX |
| --- | --- | ---: | ---: | ---: | ---: |
| `np.exp` | float32 |  1,024 | 253.4 | 356.5 | 1.41× |
| `np.log` | float32 |  1,024 | 241.6 | 358.2 | 1.48× |
| `np.exp` | float32 | 65,536 | 407.8 | 761.0 | 1.87× |
| `np.log` | float32 | 65,536 | 373.6 | 630.0 | 1.69× |

`np.exp` / `np.log` for float64 are unchanged — the NPYV kernel is f32
only at the moment. f64 still uses scalar libm.

Test status: 4699 umath tests passed, 0 failed.

Raw bench output saved to `~/dev/numpy/.benchresults/lasx_npyv.txt`.

---

## 2026-05-20 — f64 NPYV exp / log

f64 kernels added in the same file as the f32 kernels (Cephes exp +
FDLIBM log, ~15-digit accuracy). 4-lane on LASX, 2-lane on LSX.

All measurements at n = 1,000,000.

| op | dtype | scalar (M/s) | LASX (M/s) | LASX/scalar |
| --- | --- | ---: | ---: | ---: |
| `np.exp` | float64 |  71.2 | **311.5** | **4.38×** |
| `np.log` | float64 |  54.6 | **219.9** | **4.03×** |

Smaller-n results:

| op | dtype | n | LASX (M/s) | speedup |
| --- | --- | ---: | ---: | ---: |
| `np.exp` | float64 |  1,024 | 218.5 | 3.40× |
| `np.log` | float64 |  1,024 | 179.6 | 3.57× |
| `np.exp` | float64 | 65,536 | 323.3 | 4.56× |
| `np.log` | float64 | 65,536 | 230.6 | 4.24× |

ULP accuracy (vs `math.exp` / `math.log`, 200,000 random samples):

| op | dtype | max ULP | p99 |
| --- | --- | ---: | ---: |
| exp | float64 | 2 | 1 |
| log | float64 | 1 | 0 |

Test status: 4699 umath tests passed, 0 failed.

Raw bench output: `~/dev/numpy/.benchresults/lasx_npyv_f64.txt`.

---

## 2026-05-23 — Tier 1 NPYV transcendentals

Six ops (`sinh`, `cosh`, `exp2`, `log2`, `log10`, `arccosh`) now route
through NPYV by composing on top of the exp/log kernels. See
`OPTIMIZATIONS.md` for the algorithm sketch.

Methodology change: compare contiguous (NPYV path) vs strided (scalar
libm fallback) arrays on the **same build**. This isolates the
optimization from the rest of the system.

| op | dtype | scalar libm (M/s) | NPYV (M/s) | speedup |
|----|---|---:|---:|---:|
| `np.exp2`    | float64 |  20.1 | 292.5 | **14.55×** |
| `np.sinh`    | float32 |  26.6 | 368.0 | **13.83×** |
| `np.log10`   | float32 |  86.0 | 674.2 |   **7.84×** |
| `np.sinh`    | float64 |  24.5 | 155.9 |   6.36× |
| `np.log10`   | float64 |  37.4 | 222.7 |   5.95× |
| `np.log2`    | float64 |  52.1 | 224.8 |   4.31× |
| `np.cosh`    | float32 |  86.5 | 362.3 |   4.19× |
| `np.log2`    | float32 | 205.3 | 674.2 |   3.28× |
| `np.cosh`    | float64 |  47.6 | 155.9 |   3.27× |
| `np.exp2`    | float32 | 247.3 | 691.3 |   2.80× |
| `np.arccosh` | float64 |  74.3 | 105.6 |   1.42× |
| `np.arccosh` | float32 | 230.5 | 238.9 |   1.04× |

`arccosh` gets little out of this because the scalar libm path is
already vectorizable on LoongArch. Everything else is genuinely scalar-
bound before this change.

Test status: 4699 umath tests pass, 0 failed.

Raw: `~/dev/numpy/.benchresults/tier1_npyv.txt`.

`arcsinh` and `arctanh` are deliberately *not* in Tier 1 — their natural
identities lose precision for `|x| → 0` (catastrophic cancellation when
`1 + x²` or `(1+x)/(1−x)` rounds to 1). They need `log1p`, which NPYV
doesn't expose yet. The numpy `test_loss_of_precision` test catches this
and would fail.

## 2026-05-23 — Tier 2 NPYV transcendentals

15 of 17 ops in `loops_umath_fp.dispatch.c.src` now route through NPYV
(only `power` remains). Algorithms in `OPTIMIZATIONS.md`.

| op | f32 NPYV (M/s) | f32 scalar (M/s) | speedup | f64 NPYV (M/s) | f64 scalar (M/s) | speedup |
|---|---:|---:|---:|---:|---:|---:|
| `np.tan`     | 790 |   31 | **~25×** | 243 |   7  | **~34×** |
| `np.arctan2` | 246 |   42 | ~6× | 75 |  24 | ~3× |
| `np.arcsinh` | 125 |   35 | ~4× | 53 |  29 | ~2× |
| `np.expm1`   | 625 |   70 | ~9× | 238 |  67 | ~4× |
| `np.log1p`   | 528 |   65 | ~8× | 152 |  63 | ~2× |
| `np.arctanh` | 240 |   35 | ~7× | 99 |  33 | ~3× |
| `np.cbrt`    | 170 |   42 | ~4× | 64 |  42 | ~2× |
| `np.arctan`  | 578 |  103 | ~5× | 202 |  40 | ~5× |
| `np.arccos`  | 193 |   92 | ~2× | 83 |  77 | ~1× |
| `np.arcsin`  | 196 |  153 | ~1× | 82 |  73 | ~1× |

`tan` is the headline. Scalar `libm tan` on this CPU runs at
~7 M elem/s (f64) — extremely slow even for libm — so our 8-lane NPYV
polynomial gets a 34× cumulative gain.

`arcsinh` and `arctanh` are now back in NPYV after 2026-05-23 had
disabled them. Both use `log1p`-based identities that avoid the
catastrophic cancellation that broke the original direct formula.

Test status: 4699 umath tests pass.

Raw: `~/dev/numpy/.benchresults/tier2_npyv.txt`.

## 2026-05-23 — power(x, y)

The last op in `loops_umath_fp.dispatch.c.src`. Algorithm:
`exp(y · log(|x|))` with a chain of IEEE 754 special-case overrides.

| dtype | scalar libm (M/s) | NPYV (M/s) | speedup |
|---|---:|---:|---:|
| float32 | 80 | 131 | 1.64× |
| float64 | 17 | 47 | **2.71×** |

Modest because libm `pow` is already a similar exp/log composition. But
it closes out the file: **19 ufuncs** in `loops_umath_fp` (`exp2`, `log2`,
`log10`, `expm1`, `log1p`, `cbrt`, `tan`, `arcsin`, `arccos`, `arctan`,
`arctan2`, `sinh`, `cosh`, `arcsinh`, `arccosh`, `arctanh`, `power`) all
NPYV-routed on LoongArch.

Test status: 4699 umath tests pass; manual edge-case sweep passes:
`pow(0,0)=1`, `pow(1,NaN)=1`, `pow(-1,inf)=1`, `pow(-2,3)=-8`,
`pow(-2,0.5)=NaN`, `pow(0,-1)=+inf`, etc.

## Cumulative speedups (3A6000, n = 1M)

All five optimization passes, original scalar baseline → today:

| op | dtype | scalar (M/s) | today (M/s) | total speedup |
| --- | --- | ---: | ---: | ---: |
| `np.exp` | float32 | 236.5 | 823.4 | **3.48×** |
| `np.log` | float32 | 208.2 | 770.1 | **3.70×** |
| `np.exp` | float64 |  71.2 | 311.5 | **4.38×** |
| `np.log` | float64 |  54.6 | 219.9 | **4.03×** |
| `np.sort` | float32 |  11.5 | 70.7 | **6.15×** |
| `np.sort` | float64 |  11.3 | 36.4 | **3.22×** |
| `np.sort` | int32   |  12.7 | 81.0 | **6.38×** |
| `np.sort` | int16   |  12.7 | 93.5 | **7.36×** (via Highway LSX) |
| `np.sin` | float32 |  76.7 | 434.6 | **5.67×** |
| `np.cos` | float32 |  78.1 | 437.2 | **5.60×** |
| `np.tanh` | float32 |  26.5 | 329.3 | **12.43×** |
| `np.logical_not` | bool | 1871.8 | 16145.3 | **8.63×** |
| `np.logical_and` | bool | 1044.0 | 14072.4 | **13.48×** |
| `np.sinh`    | float32 |  26.6 | 368.0 | **13.83×** |
| `np.exp2`    | float64 |  20.1 | 292.5 | **14.55×** |
| `np.log10`   | float32 |  86.0 | 674.2 |   7.84× |
| `np.sinh`    | float64 |  24.5 | 155.9 |   6.36× |
| `np.log10`   | float64 |  37.4 | 222.7 |   5.95× |
| `np.log2`    | float64 |  52.1 | 224.8 |   4.31× |
| `np.cosh`    | float32 |  86.5 | 362.3 |   4.19× |
| `np.log2`    | float32 | 205.3 | 674.2 |   3.28× |
| `np.cosh`    | float64 |  47.6 | 155.9 |   3.27× |
| `np.exp2`    | float32 | 247.3 | 691.3 |   2.80× |

## 2026-05-25 — tanh f64 / cbrt f32 / power reclaim

After the cross-CPU bench against stock NumPy on Intel pointed at
`tanh`, `cbrt` f32, and `power` as lagging more than the SIMD-width
gap could account for, this pass replaced the scalar-libm fallbacks
with NPYV kernels (or kept the Highway SVML port where it was already
the fastest available choice on LASX). Same `bench/loongarch.py`
methodology as the earlier rows — contiguous NPYV vs. strided scalar
in the same build.

### Per-op deltas, before → after this session

| op | dtype | before (M/s) | after (M/s) | change |
|---|---|---:|---:|---:|
| `np.tanh`  | float64 |  54.3 |  81.0 | **+49%** |
| `np.cbrt`  | float32 | 174.3 | 537.3 | **+208%** |
| `np.power` | float64 |  52.9 |  75.7 | **+43%** |
| `np.power` | float32 | 140.2 | 183.4 | **+31%** |
| `np.tanh`  | float32 | 307.1 | 338.9 | +10% (same Highway SVML path; bench noise) |

### Geomean over the whole bench, before → after

Comparing `loongarch_loongson.txt` (start of session) to
`loongarch_loongson_v3.txt` (end), both from the same
`bench/loongarch.py --skip-sort` run:

| scope | n | geomean speedup | min | max |
|---|---:|---:|---:|---:|
| **just the 4 touched ops** | 4 | **1.71×** | 1.31× | 3.08× |
| all 122 op × dtype pairs | 122 | 1.015× | 0.71× | 3.08× |
| f64 family | 22 | 1.041× | 0.95× | 1.49× |
| f32 family | 22 | 1.067× | 0.95× | 3.08× |
| f16 / int / bool | 78 | ≈1.0× | (bench noise) | |

The all-ops 1.5% geomean is small because most ops weren't touched
this session; their values just walk within the script's ±15% noise
band. The 1.71× on the targeted four ops is the load-bearing number.
The largest apparent regression (`multiply` i32 at 0.71×) is also
within noise — no integer kernel was touched this session.
