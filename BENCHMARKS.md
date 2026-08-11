# Loongson 3A6000 benchmarks

This document records the current DragonArray performance snapshot on a
Loongson 3A6000. It does not claim results for the 3C5000.

## Test system

| | Stock NumPy | DragonArray |
|---|---|---|
| Version | `2.5.0.dev0+git20260515.79b0331` | `2.5.0.dev0+dragon.unofficial.1` |
| Source | upstream `79b033101a` | current 2026-08-10 working tree |
| Runtime SIMD | LSX baseline | LSX baseline + LASX dispatch |

- CPU: Loongson 3A6000 at 2.0 GHz
- OS: Linux 5.4.18-167-generic, loongarch64
- Compiler: GCC 15.2.0
- Python: 3.13.13
- Benchmark: `bench/loongarch.py --iters 30`
- Array length: 1,000,000 elements

Throughput is reported in millions of contiguous elements per second. Speedup
is DragonArray throughput divided by stock NumPy throughput. Both builds were
run on the same machine with the same benchmark inputs.

## Summary

The full run covers 135 operation-and-dtype pairs. Its raw geometric-mean
speedup is **1.81x**.

| Family | Pairs | Raw geomean | Pairs at least 1.10x |
|---|---:|---:|---:|
| Boolean | 2 | 8.31x | 2 |
| `float16` | 20 | 2.77x | 9 |
| `float32` | 27 | 1.61x | 8 |
| `float64` | 27 | 1.25x | 6 |
| Integer | 59 | 1.86x | 32 |
| **All** | **135** | **1.81x** | **57** |

The raw aggregate includes the benchmark's fixed operation order. Basic
floating-point arithmetic is order-sensitive on this machine, so its confirmed
results are reported separately below.

## Largest current gains

| Operation | Dtype | Stock | DragonArray | Speedup |
|---|---|---:|---:|---:|
| `arctanh` | `float16` | 24.0 | 453.9 | 18.91x |
| `floor_divide` | `int8` | 163.5 | 2675.7 | 16.37x |
| `mod` | `int8` | 173.7 | 2475.5 | 14.25x |
| `sin` | `float16` | 52.4 | 689.3 | 13.16x |
| `cos` | `float16` | 51.4 | 662.0 | 12.88x |
| `arcsinh` | `float16` | 21.1 | 261.2 | 12.38x |
| `tanh` | `float32` | 26.5 | 319.0 | 12.04x |
| `floor_divide` | `int16` | 157.7 | 1827.1 | 11.59x |
| `log1p` | `float16` | 36.1 | 392.4 | 10.87x |
| `logical_and` | `bool` | 1044.5 | 11346.8 | 10.86x |

## Floating-point kernels

These are the floating-point operation-and-dtype pairs that currently measure
at least 1.10x faster than stock. Unlisted transcendental pairs are at parity in
this run.

| Dtype | Operation | Stock | DragonArray | Speedup |
|---|---|---:|---:|---:|
| `float16` | `arcsinh` | 21.1 | 261.2 | 12.38x |
| `float16` | `arctanh` | 24.0 | 453.9 | 18.91x |
| `float16` | `cbrt` | 36.1 | 145.7 | 4.04x |
| `float16` | `cos` | 51.4 | 662.0 | 12.88x |
| `float16` | `log` | 68.0 | 442.1 | 6.50x |
| `float16` | `log10` | 43.2 | 421.8 | 9.76x |
| `float16` | `log1p` | 36.1 | 392.4 | 10.87x |
| `float16` | `log2` | 67.8 | 422.7 | 6.24x |
| `float16` | `sin` | 52.4 | 689.3 | 13.16x |
| `float32` | `arcsin` | 95.9 | 194.1 | 2.02x |
| `float32` | `arctan` | 64.2 | 573.1 | 8.93x |
| `float32` | `arctan2` | 43.5 | 205.7 | 4.73x |
| `float32` | `cos` | 78.1 | 410.5 | 5.26x |
| `float32` | `sin` | 77.4 | 423.4 | 5.47x |
| `float32` | `tanh` | 26.5 | 319.0 | 12.04x |
| `float64` | `arctan` | 45.4 | 207.5 | 4.57x |
| `float64` | `arctan2` | 24.6 | 80.5 | 3.27x |
| `float64` | `cbrt` | 42.8 | 136.7 | 3.19x |
| `float64` | `log1p` | 63.4 | 262.8 | 4.15x |
| `float64` | `tanh` | 24.0 | 54.8 | 2.28x |

`float32` divide is listed with the other basic arithmetic results because it
was checked under both benchmark orders.

## Basic floating-point arithmetic

The table reports median DragonArray/stock ratios from matched rounds in two
confirmation runs. "Dtype grouped" measures all four operations for one dtype
before moving to the next dtype. "Operation grouped" alternates `float32` and
`float64` for each operation.

| Operation | f32 dtype grouped | f32 operation grouped | f64 dtype grouped | f64 operation grouped |
|---|---:|---:|---:|---:|
| `add` | 0.99x | 0.83x | 0.99x | 0.71x |
| `subtract` | 0.99x | 1.05x | 1.00x | 0.73x |
| `multiply` | 1.00x | 1.05x | 1.01x | 0.73x |
| `divide` | 1.91x | 1.92x | 1.13x | 0.78x |

`float32` divide is the stable result here: it remains about **1.9x** faster in
both orders. The other arithmetic figures should be treated as
working-set/order-sensitive measurements, not stable regressions or gains.

## Integer, boolean, and sort highlights

LASX integer division is most effective through 32 bits. Integer modulo is
faster for every measured width.

| Operation | Dtype | Stock | DragonArray | Speedup |
|---|---|---:|---:|---:|
| `floor_divide` | `int8` | 163.5 | 2675.7 | 16.37x |
| `floor_divide` | `int16` | 157.7 | 1827.1 | 11.59x |
| `floor_divide` | `int32` | 165.3 | 1114.8 | 6.74x |
| `floor_divide` | `uint8` | 408.9 | 2425.8 | 5.93x |
| `floor_divide` | `uint16` | 286.2 | 1744.9 | 6.10x |
| `floor_divide` | `uint32` | 189.9 | 1117.4 | 5.88x |
| `mod` | `int8` | 173.7 | 2475.5 | 14.25x |
| `mod` | `int16` | 168.3 | 1641.7 | 9.76x |
| `mod` | `int32` | 157.1 | 1052.7 | 6.70x |
| `mod` | `int64` | 102.8 | 527.0 | 5.13x |
| `logical_and` | `bool` | 1044.5 | 11346.8 | 10.86x |
| `logical_not` | `bool` | 1866.3 | 11875.2 | 6.36x |
| `sort` | `float32` | 11.0 | 74.9 | 6.81x |
| `sort` | `float64` | 11.0 | 36.8 | 3.35x |
| `sort` | `int16` | 14.7 | 93.6 | 6.37x |
| `sort` | `int32` | 14.8 | 80.3 | 5.43x |
| `sort` | `int64` | 15.6 | 41.8 | 2.68x |

Integer `max` reductions measure between 1.28x and 2.00x across the eight
signed and unsigned integer widths. Integer add, multiply, comparison, and
argmax are generally near parity at this array size.

## Reproducing the run

Run the two builds back-to-back on an otherwise idle 3A6000:

```bash
PYTHONPATH=<stock-install> \
    python3 bench/loongarch.py --iters 30 > stock.txt
PYTHONPATH=<dragon-install> \
    python3 bench/loongarch.py --iters 30 > dragon.txt
```

Compare the contiguous-throughput column row by row. The benchmark performs
three warm-up calls before each timed loop. Sort uses 10 timed iterations;
other operations use 30.

The exact 2026-08-10 artifacts are:

- [stock raw output](.benchresults/2026-08-10-stock-loongarch-raw.txt)
- [DragonArray raw output](.benchresults/2026-08-10-dragon-loongarch-raw.txt)
- [stock parsed CSV](.benchresults/2026-08-10-before-stock-2.5.0.csv)
- [DragonArray parsed CSV](.benchresults/2026-08-10-after-dragon-2.5.0.csv)
- [DragonArray/stock comparison](.benchresults/2026-08-10-dragon-vs-stock-2.5.0.csv)
- [dtype-grouped arithmetic confirmation](.benchresults/2026-08-10-arithmetic-confirm-dtype-order.txt)
- [operation-grouped arithmetic confirmation](.benchresults/2026-08-10-arithmetic-confirm-op-order.txt)
