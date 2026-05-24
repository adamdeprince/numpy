# Benchmarks — dragon-array vs. stock NumPy 2.5.0

The only comparison that matters for a release is this build vs. the NumPy it
derives from, on the target hardware. Everything here is:

> dragon-array `2.5.0.dev0+dragon.unofficial.1` vs stock NumPy `2.5.0.dev0`,
> both built from the same source tree with the same toolchain
> (`/opt/loongson-gcc-15.2.0`), on a Loongson 3A6000 @ 2.0 GHz.

Numbers are M elements/sec on the contiguous path — each build running its
*best* available SIMD (stock uses LSX + Highway; dragon adds LASX). The ratio is
the speedup a user gets by installing this wheel instead of building stock NumPy
on the same chip.

Earlier development-snapshot numbers (vs. scalar fallback, vs. intermediate
builds) have been removed — a reader can't reproduce those, only this
release-vs-stock comparison.

## Headline

Geometric mean 3.31× across 135 operation × dtype pairs. Range 0.81×–26×.

| dtype family | ops | geomean |
|---|---:|---:|
| float16 | 20 | 10.72× |
| bool | 2 | 11.43× |
| float32 | 27 | 4.46× |
| float64 | 27 | 3.60× |
| integer | 59 | 1.79× |
| all | 135 | 3.31× |

float16 and the float transcendentals see the largest gains because stock NumPy
falls back to scalar libm for them on LoongArch — dragon vectorizes them.
Integer is lower because stock already vectorizes most integer ops on LSX, so
dragon is beating LSX (or matching it on bandwidth-bound ops), not scalar.

## Highlights

### float16 (LASX f16↔f32 bridge — stock has no SIMD here)
| op | stock | dragon | speedup |
|---|---:|---:|---:|
| `tanh` | 21 | 542 | 26.1× |
| `sinh` | 11 | 216 | 19.0× |
| `expm1` | 36 | 637 | 18.0× |
| `tan` | 29 | 463 | 16.3× |
| `arctanh` | 24 | 367 | 15.7× |
| `log10` | 43 | 635 | 14.8× |

(20 f16 ops, geomean 10.7×.)

### Transcendentals (speedup, f64 / f32)
| op | f64 | f32 |
|---|---:|---:|
| `exp` | 4.74× | 3.32× |
| `log` | 4.45× | 3.77× |
| `sin` | 3.49× | 5.75× |
| `cos` | 3.51× | 5.70× |
| `tanh` | 5.76× | 13.0× |
| `cbrt` | 3.21× | 10.5× |
| `exp2` | 14.6× | 2.94× |
| `arctanh` | 14.8× | 7.80× |
| `power` | 3.54× | 2.21× |

### Integer divide / modulo (native LASX `xvdiv`/`xvmod`; stock = scalar)
| op | dtype | speedup |
|---|---|---:|
| `floor_divide` | int8 | 16.7× |
| `mod` | int8 | 14.3× |
| `mod` | i32 | 6.70× |
| `floor_divide` | i32 | 6.69× |
| `floor_divide` | uint8 | 5.93× |

### Sort (Highway qsort, now LASX)
| dtype | stock | dragon | speedup |
|---|---:|---:|---:|
| f32 | 11 | 72 | 6.5× |
| i16 | 15 | 93 | 6.3× |
| i32 | 15 | 78 | 5.2× |
| f64 | 11 | 36 | 3.2× |
| i64 | 16 | 40 | 2.6× |

### Boolean & minmax
| op | dtype | speedup |
|---|---|---:|
| `logical_and` | bool | 14.8× |
| `logical_not` | bool | 8.8× |
| `max` | i64 | 1.35× |
| `max` | uint8 | 1.27× |

## Tradeoff: f32 `add` / `subtract` / `multiply` regress

| op | f32 | f64 |
|---|---:|---:|
| `divide` | 1.92× | 1.04× |
| `add` | 0.81× | 1.14× |
| `subtract` | 0.81× | 1.10× |
| `multiply` | 0.81× | 1.09× |

Measured: f32 `add`/`subtract`/`multiply` run ~19% slower on LASX than on
stock's LSX at n=1,000,000, while f32 `divide` and all f64 arithmetic are
faster. `loops_arithm_fp` is one dispatch unit (no per-op split), so it's kept
on LASX for the `divide`/f64 wins; large-array f32 `add`/`mul` are faster on
stock today.

Cause not profiled — the likely explanation is that these are
memory-bandwidth-bound and 256-bit streaming isn't faster than 128-bit here,
but that's a hypothesis, not measured. Whether it differs on cache-resident
data or higher-bandwidth parts is untested.

Comparison ops (`less`/`greater`/…) are kept on LSX: measured 0.78–0.92× on
LASX for 32/64-bit dtypes (8/16-bit flat), with no dtype above 1.02×. The LASX
path for their 1-byte bool output uses cross-128-bit-lane packing the LSX path
avoids (visible in the backend code) — the likely cause, given the regression
tracks pack depth.

## Methodology

- Both numpys built from the same tree, same `gcc-15`, libstdc++ statically
  linked. Versions: `2.5.0.dev0` (stock) and `2.5.0.dev0+dragon.unofficial.1`.
- `bench/loongarch.py --iters 30`, n = 1,000,000, run back-to-back on an
  otherwise-idle box — the only valid way. Comparing runs taken minutes apart
  under different load produces garbage; watch the `max`/`multiply` *control*
  ops (where the result should be stable) to catch a contaminated run.
- Reproduce:
  ```bash
  PYTHONPATH=<stock-install>  python bench/loongarch.py --iters 30 > stock.txt
  PYTHONPATH=<dragon-install> python bench/loongarch.py --iters 30 > dragon.txt
  # then compare the contiguous column op-by-op
  ```

## Hardware

- CPU: Loongson 3A6000 @ 2.0 GHz, LSX (128-bit) + LASX (256-bit) in HWCAP.
- Toolchain: GCC 15.2.0 (`/opt/loongson-gcc-15.2.0`).
- Python: 3.13.
