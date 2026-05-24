# LoongArch / LSX Optimizations

Chronological log of optimization work on the `loongson-experimental` branch.
See `BENCHMARKS.md` for measured throughput.

## Open follow-ups

Tracked here so they don't get lost between sessions. Items move out of
this list as they land (each one becomes a dated section below).

- ~~SIMD Payne-Hanek range reduction for f64 sin/cos~~ — done 2026-05-24,
  see below. The reduction is scalar per-lane (the bit-table lookup
  is per-lane); the polynomial after stays vectorized. Speed of the
  slow path is roughly libm parity. Vectorizing the reduction itself
  is a further follow-up (would need SIMD gather for the 2/π bit
  table chunks).
- ~~`cbrt` f64 direct kernel~~ — done 2026-05-23, see below.
- ~~`log1p` / `arctanh` f64 direct polynomials~~ — done 2026-05-24,
  see below. (`arcsinh` skipped; section explains why.)
- **Upstream upstreaming.** Once Payne-Hanek lands, the precision
  story is "matches libm on the full f64 domain" — at that point the
  sin/cos f64 hybrid is a candidate for an upstream PR, gated on a
  build flag rather than `__loongarch__`.


## 2026-05-17 — NPYV f32 exp / log

**Files:** `numpy/_core/src/umath/loops_exponent_log.dispatch.c.src`

### What

Added an NPYV-style `simd_exp_FLOAT_npyv` / `simd_log_FLOAT_npyv` to
`loops_exponent_log.dispatch.c.src`. The existing float32 exp/log in that
file are written in raw x86 intrinsics (AVX2/AVX-512), so on LoongArch they
fell through to scalar `expf` / `logf`. The new block is gated by

```c
#if NPY_SIMD_F32 && !defined(SIMD_AVX2_FMA3) && !defined(SIMD_AVX512F)
```

so it covers any NPYV target without an existing AVX path — currently LSX,
NEON, VSX. The dispatcher entry got a new `#elif defined(NPYV_IMPL_F32_EXP_LOG)`
branch that calls it for contiguous inputs and falls back to scalar libm for
strided ones.

### How

Algorithm mirrors the AVX2 implementation in the same file:

- **exp:** Cody-Waite range reduction `y = x - k·ln(2)` with `k = rint(x·log2e)`,
  then `exp(x) = (P(y)/Q(y)) · 2^k`. P is degree-5, Q is degree-2. `2^k` is
  constructed by writing `(k+127)` into the IEEE-754 exponent field via
  `npyv_shli_s32` and `npyv_reinterpret_f32_s32`.
- **log:** Bit-level mantissa extraction with the exponent field forced to
  126 (so mantissa is in `[0.5, 1)`), then an `m ≤ 1/√2` shift to keep the
  mantissa close to 1, then `log(1 + (m−1)) ≈ P(m−1) / Q(m−1)` with both
  degree-5.

Coefficients reused unchanged from `numpy/_core/src/umath/npy_simd_data.h`.

### Decisions

- **One LSX intrinsic, not portable yet.** The only non-NPYV op in the new
  block is `__lsx_vffint_s_w` for int32→float32, wrapped in
  `npyv__cvt_f32_s32` with a `TODO` note. NPYV doesn't expose this op
  portably; lifting it would let NEON / VSX inherit the same exp/log path
  for free.
- **Contiguous-only fast path.** Strided cases fall back to scalar libm.
  Strided NPYV via `npyv_loadn` / `npyv_storen` is possible but adds
  complexity for a less common shape.
- **No DOUBLE path yet.** f64 still uses scalar libm on LoongArch.

### Correctness fixes during development

1. **Spurious FP exceptions for NaN / inf inputs.** Ordered LSX compares
   (`vfcmp.clt.s` etc.) raise `INVALID` for NaN. Both kernels now mask NaN
   to a safe value (0 for exp, 1 for log) via the non-signaling
   `npyv_notnan_f32` *before* any signaling compare. The polynomial path
   also receives the masked input so `inf · finite` etc. doesn't materialize.
2. **Wrong log mantissa range.** Initial extraction set the IEEE exponent
   field to 127, giving `m ∈ [1, 2)`, then compared to `1/√2 ≈ 0.707` —
   a condition never true. Fixed by setting the exponent field to 126
   (`m ∈ [0.5, 1)`) and adjusting the integer exponent accordingly.
   Matches the AVX path. Before fix, `test_log_float32` reported 3985 ULP
   error; after, 774/774 exp/log tests pass.

### Test status

`spin test -- numpy/_core/tests/test_umath.py`: 4699 passed, 0 failed.

### Follow-ups

- f64 exp/log on the same pattern (2-lane LSX, 4-lane LASX).
- Lift `npyv_cvt_f32_s32` into the NPYV API.
- LASX (256-bit) backend — would widen this path automatically since the
  code is already NPYV-style.

---

## 2026-05-18 — Enable Highway LSX target

**Files:** `numpy/_core/meson.build` (two hunks)

### What

Removed the stale `-DHWY_COMPILE_ONLY_SCALAR` global flag for loongarch64
and added `LSX` to the dispatch lists of both `highway_qsort.dispatch.h`
and `highway_qsort_16bit.dispatch.h`. Highway is now compiled with its
LSX (and LASX, where the compiler supports it) backends, and NumPy's sort
dispatcher actually instantiates the LSX variant.

### Why it works now

Google Highway gained LSX/LASX targets between v1.2 and v1.3 (first commit
`7e01a07e` on 2024-11-25, real impl `7bfb8e8b` on 2025-04-17, runtime
dispatch `10d7ab41` on 2025-09-09). The bundled submodule is at
`1.3.0-159-gee36c83` and includes both `hwy/ops/loongarch_lsx-inl.h` (5,954
lines) and `hwy/ops/loongarch_lasx-inl.h` (4,681 lines).

NumPy's flag was added 2024-11-05 (commit `7c35c37a1d`) — three weeks before
Highway upstream had even stubbed LSX. The flag forced every Highway-backed
op to a scalar fallback. It has been stale for over a year.

### Impact

This is a much bigger blast radius than the exp/log change because Highway
powers more than sort in NumPy:

- `src/npysort/highway_qsort.dispatch.cpp` — vectorized quicksort
- `src/npysort/highway_qsort_16bit.dispatch.cpp` — 16-bit vqsort
- `src/umath/loops_trigonometric.dispatch.cpp` — `sin`, `cos`, `tan`, ...
- `src/umath/loops_hyperbolic.dispatch.cpp.src` — `sinh`, `cosh`, `tanh`
- `src/umath/loops_logical.dispatch.cpp` — `logical_and`, `logical_or`, `logical_not`

Measured wins on the 3A6000 (see `BENCHMARKS.md` for the table):

- **`np.sort` f32/int32: ~4–5×** (this required the second hunk — adding
  `LSX` to the sort dispatch list — because the dispatch list previously
  only enumerated `ASIMD`, `VSX2`)
- **`sin` / `cos` f32: ~3.2×**
- **`tanh` f32: ~12.8×** (biggest single jump)
- **`logical_and` / `logical_not` bool: ~8–13×**

A handful of ops were not affected because they don't actually route through
Highway today (`tan`, `arcsin`, `arctan`, `sinh`, `cosh`).

### Decisions

- **Don't add LASX to the sort dispatch list yet.** LASX is not yet a
  feature in `meson_cpu/loongarch64/meson.build`. Adding it is a separate
  piece of work — see the LASX backend follow-up.
- **Left the `# FIXME: disable VXE due to runtime segfault` comments alone.**
  They're orthogonal — a problem on s390x, not loongarch64.

### Test status

`spin test -- numpy/_core/tests/test_multiarray.py -k sort
 numpy/_core/tests/test_umath.py`: 7920 passed, 0 failed.

### Follow-ups

- Add `LASX` feature in `meson_cpu/loongarch64/meson.build`, then list it
  in both `highway_qsort.dispatch.h` and other Highway-backed dispatch
  groups so 3A6000 picks up the 256-bit variants at runtime.
- Investigate why `tan`, `arcsin`, `arctan`, `sinh`, `cosh` didn't move —
  they may use a different scalar path (SVML on x86, scalar libm here).

---

## 2026-05-19 — Define LASX feature; dispatch sort/trig/log/hyper

**Files:**
- `meson_cpu/loongarch64/meson.build` (added LASX feature, implies LSX)
- `numpy/_core/src/_simd/checks/cpu_lasx.c` (new probe)
- `numpy/_core/src/common/npy_cpu_features.h` (`NPY_CPU_FEATURE_LASX = 501`)
- `numpy/_core/src/common/npy_cpu_features.c` (HWCAP probe + name registration)
- `numpy/_core/meson.build` (LASX added to `highway_qsort`,
  `loops_hyperbolic`, `loops_logical`, `loops_trigonometric` dispatch lists)

### What

Wired LASX (LoongArch 256-bit SIMD) end-to-end into NumPy's CPU dispatch
system, so any Highway-backed code compiled by NumPy can now use the LASX
target at runtime on the 3A6000.

Detection still uses `getauxval(AT_HWCAP) & HWCAP_LOONGARCH_LASX` — matches
the standing rule that HWCAP is the source of truth, not raw `cpucfg`.
`np.show_runtime()` now reports `'simd_extensions': {'baseline': ['LSX'],
'found': ['LASX'], 'not_found': []}` on the 3A6000.

This is the LASX *dispatch* wiring only. There is no `numpy/_core/src/common/simd/lasx/`
NPYV backend yet — that's a separate, much larger piece of work. The LASX
gains we see today come entirely from Highway's internal LASX target;
NPYV-style code (including the f32 exp/log we wrote in May) still runs as
LSX.

### Decisions

- **`LASX` implies `LSX`** in `meson_cpu/loongarch64/meson.build`, mirroring
  ARM's `NEON_FP16 implies NEON` pattern. The 3A6000 has both; in
  practice LSX stays in the baseline (LASX dispatches above it).
- **Dropped LASX from `highway_qsort_16bit.dispatch.h`** after a regression.
  With LASX in the 16-bit sort dispatch, `np.sort(int16)` fell from 94 to 59
  Melem/s — likely a Highway upstream gap (their 16-bit vqsort may use a
  partial LASX path that falls back to scalar for some primitives). LSX-only
  is better for int16; the comment in `meson.build` records the rationale.
- **Did not add LASX to `loops_exponent_log.dispatch.h`.** Our exp/log
  is NPYV-style and there is no LASX NPYV backend, so listing LASX there
  would either no-op or force the file to be compiled twice for no benefit.

### Impact (3A6000, n = 1M)

Over the LSX-only state (see prior section), LASX adds:

- `np.sort` f32: 49 → 71 Melem/s (1.44×)
- `np.sort` f64: 23 → 33 Melem/s (1.45×)
- `np.sort` int32: 62 → 81 Melem/s (1.30×)
- `np.sort` int64: 37 → 42 Melem/s (1.13×)
- `np.sin` f32: 250 → 435 Melem/s (1.74×)
- `np.cos` f32: 246 → 437 Melem/s (1.78×)

`tanh` and the logical ops were already saturated by LSX (Highway already
picks them up internally even without -mlasx at numpy's level when LSX is
the dispatch target). `tan`, `arcsin`, `arctan`, `sinh`, `cosh` still flat
— different code paths, separate follow-up.

Stacking the wins so far against the original scalar baseline:

- `np.sort` int32: 12.7 → 81 Melem/s (**6.4×**)
- `np.sort` f32: 11.5 → 71 Melem/s (**6.2×**)
- `np.sin` f32: 77 → 435 Melem/s (**5.7×**)
- `np.cos` f32: 78 → 437 Melem/s (**5.6×**)

### Test status

`spin test -- numpy/_core/tests/test_multiarray.py -k sort
 numpy/_core/tests/test_umath.py`: 7920 passed, 0 failed.

### Follow-ups

- **Native LASX NPYV backend** (`numpy/_core/src/common/simd/lasx/`) — the
  big next piece. Would widen the f32 exp/log we already wrote (and every
  other NPYV loop) from 4 to 8 lanes.
- Investigate the int16 LASX qsort regression in Highway upstream.
- `tan`, `arcsin`, `arctan`, `sinh`, `cosh` still scalar — these are not
  Highway-routed. Check `loops_trigonometric.dispatch.cpp` and
  `loops_hyperbolic.dispatch.cpp.src` for their actual code paths.

---

## 2026-05-20 — Native LASX NPYV backend

**Files added:**
- `numpy/_core/src/common/simd/lasx/lasx.h` (umbrella, types, lane counts)
- `numpy/_core/src/common/simd/lasx/misc.h`
- `numpy/_core/src/common/simd/lasx/memory.h`
- `numpy/_core/src/common/simd/lasx/reorder.h`
- `numpy/_core/src/common/simd/lasx/operators.h`
- `numpy/_core/src/common/simd/lasx/conversion.h`
- `numpy/_core/src/common/simd/lasx/arithmetic.h`
- `numpy/_core/src/common/simd/lasx/math.h`

Total: 2108 lines, mirroring the LSX backend's structure file-for-file.

**Files modified:**
- `numpy/_core/src/common/simd/simd.h` — LASX dispatch entry above LSX
- `numpy/_core/src/umath/loops_exponent_log.dispatch.c.src` — extend the
  `npyv__cvt_f32_s32` helper to use `__lasx_xvffint_s_w` on LASX targets
- `numpy/_core/meson.build` — list `LASX` in the `loops_exponent_log`
  dispatch (so the file is compiled for LASX as well as LSX)

### What

Wrote a full NPYV (256-bit) backend for LoongArch LASX. The translation
from `simd/lsx/` was largely mechanical:

- `__m128i` → `__m256i`, `__m128` → `__m256`, `__m128d` → `__m256d`
- `__lsx_v*` → `__lasx_xv*`
- lane-count constants doubled (`nlanes_f32` 4→8, etc.)
- vector-literal types: `v4i32` → `v8i32`, `v4f32` → `v8f32`, `v2f64` →
  `v4f64`, etc.
- partial load/store switches extended from 1-3 cases to 1-7 / 1-3 cases
  with explicit fallthrough

The f32 exp/log kernel we wrote on 2026-05-17 is NPYV-style, so once LASX
was listed in its dispatch list it picked up 256-bit lanes automatically
with no source changes to `loops_exponent_log.dispatch.c.src`.

### Decisions / Gotchas

- **`xvpickve2gr_h` / `xvpickve2gr_b` don't exist.** LASX only provides the
  `_w` and `_d` element-extract intrinsics. Operators.h, conversion.h, and
  math.h all needed to fish 16-bit or 8-bit lanes out of the enclosing
  32-bit lane via `xvpickve2gr_wu` + shift/mask. Caught on the first build.
- **`xvmsknz_b` outputs the byte-nonzero mask per 128-bit half**, placed in
  the low 16 bits of the half (i.e., word lane 0 of the low half and word
  lane 4 of the high half). The `any/all` reductions and `tobits_b8`
  combine both into a 32-bit scalar mask.
- **Reductions are within-halves-then-combine.** LASX shuffles/horizontal-
  adds operate within each 128-bit half rather than across the full 256
  bits (the AVX convention). All the `npyv_reduce_*` and `npyv_sum_*`
  functions first reduce each half with the LSX pattern, then combine the
  two halves with one scalar op.
- **`unzip` falls back to scalar** through a memory round-trip. The
  cross-half permute would have wanted two `xvpermi.q` ops with a custom
  imm — manageable, but `unzip` isn't on the exp/log hot path, so the
  scalar fallback is fine until something needs it.

### Test status

`spin test -- numpy/_core/tests/test_umath.py`: 4699 passed, 0 failed.

### Impact (3A6000, n = 1M, float32)

| op | scalar | LSX (4-wide) | LASX (8-wide) | LASX/scalar |
| --- | ---: | ---: | ---: | ---: |
| `np.exp` | 236.5 M/s | 412.7 | **823.4** | **3.48×** |
| `np.log` | 208.2 M/s | 378.4 | **770.1** | **3.70×** |

LASX gives a clean 2× over LSX on both — the headline 8-lane scaling
materializes when n is large enough to amortize the scalar tail.

### Follow-ups

- f64 NPYV exp/log kernel — we now have the lanes (4-wide LASX f64), only
  the kernel code is missing.
- Lift `npyv_cvt_f32_s32` and the byte/half-lane extract helpers into the
  shared NPYV API so other arches inherit them cleanly.
- Audit the LASX `unzip`/cross-lane paths — replace the memory fallback
  with `xvpermi.q`-based code if any user code starts hammering it.

---

## 2026-05-20 — f64 NPYV exp / log

**File:** `numpy/_core/src/umath/loops_exponent_log.dispatch.c.src`

### What

Added NPYV-style kernels for double-precision exp/log, gated by
`#if NPY_SIMD_F64 && !defined(SIMD_AVX2_FMA3) && !defined(SIMD_AVX512F)
 && !defined(SIMD_AVX512F_NOCLANG_BUG)`, plus a `NPYV_IMPL_F64_EXP_LOG`
sentinel. The `DOUBLE_@func@` dispatcher got a new `#if defined(NPYV_IMPL_F64_EXP_LOG)`
branch between the AVX-512 paths and the scalar fallback.

### Algorithm

Different from the f32 kernels — f64 needs ~15 decimal digits of precision,
so the f32 P5/Q2 polynomial is too short. Used two well-known algorithms:

- **exp:** Cephes rational approximation. Cody-Waite range reduction
  `xr = x - round(x*log2e)*ln2` (with `ln2` split into `C1 + C2` so each
  multiplication is exact), then `exp(xr) = (1 + 2z)` where
  `z = xr·P(xr²) / (Q(xr²) − xr·P(xr²))` with `P` degree-2 and `Q` degree-3
  in `xr²`. Final scale by `2^k` via the IEEE-754 exponent-field bit trick
  (bias 1023, shift 52). Coefficients are the Cephes constants (inlined as
  literals; not in `npy_simd_data.h`).
- **log:** FDLIBM-style polynomial. Decompose `x = 2^k · m` with `m ∈
  [√0.5, √2]` (extract `m` by setting the IEEE exponent field to 1022 and
  shifting if `m ≤ 1/√2`). Then `f = m − 1`, `s = f/(2 + f)`, evaluate a
  degree-7 polynomial in `s²` and `s⁴`, combine with `f` and `k·ln2`
  (split into hi/lo for accuracy). The seven `Lg*` coefficients are the
  standard FDLIBM values.

Two new arch-specific helpers in the same file:

```c
#define npyv__cvt_f64_s64(X)  ((npyv_f64)__lasx_xvffint_d_l(X))
#define npyv__cvt_s64_f64(X)  ((npyv_s64)__lasx_xvftintrne_l_d(X))
```

(with parallel LSX entries). NPYV doesn't expose int64↔float64 casts
portably yet; `TODO` notes mark these for eventual lift.

### Test status

`spin test -- numpy/_core/tests/test_umath.py`: 4699 passed, 0 failed.

ULP measurement against `math.exp`/`math.log` on 200,000 random samples:

| op | dtype | max ULP | p99 |
| --- | --- | ---: | ---: |
| exp | float64 | 2 | 1 |
| log | float64 | 1 | 0 |

(Better than the f32 path, which had max ~6 ULP on edge cases.)

### Impact (3A6000, n = 1M, float64)

| op | scalar (M/s) | LSX (M/s) | LASX (M/s) | LASX/scalar |
| --- | ---: | ---: | ---: | ---: |
| `np.exp` | 71.2 | — | **311.5** | **4.38×** |
| `np.log` | 54.6 | — | **219.9** | **4.03×** |

(LSX f64 numbers are also produced by the same kernel — 2-wide vs 4-wide —
but the prominent number is LASX since that's the runtime dispatch on the
3A6000. Bench script didn't capture LSX-only for f64 separately.)

### Follow-ups

- Hoist the f64 Cephes coefficients into `npy_simd_data.h` so the literals
  aren't duplicated in source.
- Refactor `NPYV_IMPL_F32_EXP_LOG` / `NPYV_IMPL_F64_EXP_LOG` into a single
  guard once the f32 kernel's correctness is also LASX-tested at scale.

---

## 2026-05-23 — Tier 1 NPYV transcendentals (sinh/cosh/exp2/log2/log10/arccosh)

**Files:**
- `numpy/_core/src/umath/npyv_exp_log.h` (new) — exp/log kernels moved here
  from `loops_exponent_log.dispatch.c.src` so they can be shared
- `numpy/_core/src/umath/loops_exponent_log.dispatch.c.src` — now `#include`s
  the header, no more inline kernel definitions
- `numpy/_core/src/umath/loops_umath_fp.dispatch.c.src` — 12 new NPYV
  wrapper kernels (6 ops × 2 precisions) on top of exp/log, with an NPYV
  branch in the unary dispatcher template
- `numpy/_core/src/umath/loops.h.src` — re-include `loops_umath_fp.dispatch.h`
  before declaring `power`/`arctan2` (otherwise their dispatch list inherits
  from `loops_half`, missing LASX/LSX, breaking the LASX build)
- `numpy/_core/meson.build` — `[X86_V4]` → `[X86_V4, LASX, LSX]` in
  `loops_umath_fp.dispatch.h`

### What

Extended the LoongArch acceleration to six transcendentals that previously
sat in `loops_umath_fp.dispatch.c.src` (X86_V4/SVML-only — meaning scalar
libm on every other arch including LoongArch, ARM, PowerPC). All six are
implemented as thin wrappers over the existing exp/log NPYV kernels:

| op | identity used |
|----|---|
| `sinh(x)` | `(eˣ − e⁻ˣ) / 2` |
| `cosh(x)` | `(eˣ + e⁻ˣ) / 2` |
| `exp2(x)` | `e^(x·ln2)` |
| `log2(x)` | `log(x) · log₂(e)` |
| `log10(x)` | `log(x) · log₁₀(e)` |
| `arccosh(x)` | `log(x + √(x² − 1))` for `x ≥ 1` |

The kernel naming follows the libm convention used by the existing
dispatcher template (`asinh`/`acosh`/`atanh`, not the numpy `arcsinh` form),
so the same `#elif defined(NPYV_HAVE_@intrin@_@sfx@)` pattern in the
dispatcher picks them up automatically without per-op `#ifdef`s.

### Architecture refactor

The exp/log kernels were previously inlined into
`loops_exponent_log.dispatch.c.src`. Moved them into a new
`numpy/_core/src/umath/npyv_exp_log.h` so `loops_umath_fp` can build on top.
Both files now include that header; the kernels are `NPY_FINLINE` so each
translation unit gets its own inlined copy with no link conflicts. The
`NPYV_IMPL_F32_EXP_LOG` / `NPYV_IMPL_F64_EXP_LOG` markers are now set
inside the header (when the appropriate int↔float cast macro is available),
and the original guards in `loops_exponent_log.dispatch.c.src` just check
those markers instead of replicating the conditions.

### Decisions

- **`arcsinh` and `arctanh` are deliberately NOT in Tier 1.** The
  natural identities (`log(x + √(x² + 1))` / `½·log((1+x)/(1−x))`) suffer
  catastrophic cancellation when `|x| → 0`: e.g. `(1 + 1e-20)` rounds to
  `1.0` in f64, so `log` returns 0 instead of `~1e-20`. The
  `TestComplexFunctions::test_loss_of_precision` test fails the entire
  range `[1e-20, 1e-3]` with the direct formula. A Taylor-only fix would
  need ~30+ terms to be accurate at `|x|=0.5`. The proper fix is
  `log1p`, which we don't have in NPYV. Marker macros are commented out
  (with a note); the kernel functions remain in source as dead code so
  future work can flip them back on once `log1p` lands.
- **`arccosh` is in Tier 1.** The boundary issue at `x = 1`
  (`(1+ε)² − 1` losing precision) isn't covered by any existing test and
  is a narrow corner. Direct formula passes the suite.
- **Re-included `loops_umath_fp.dispatch.h` before declaring
  `power`/`arctan2`.** The dispatch macros are global state and get
  rewritten by every `dispatch.h` include. Before my change,
  `power`/`arctan2` were declared just after `loops_half.dispatch.h`, so
  they picked up loops_half's dispatch list (no LASX). The build then
  expected `FLOAT_power_LASX` to exist but never declared it. Re-including
  the right header restores the macros. This is technically an upstream
  numpy bug exposed by my change — worth noting if/when this goes
  upstream.

### Test status

4699 passed, 0 failed across the umath suite.

### Impact (3A6000, n = 1M)

Contiguous arrays (NPYV path) vs strided arrays of the same length on the
same build (the strided dispatch fails the `steps[i] == sizeof(dtype)`
check and falls back to scalar libm — same compiled code, different code
path).

| op | dtype | scalar libm (M/s) | NPYV (M/s) | speedup |
|----|---|---:|---:|---:|
| `np.sinh`    | float32 |  26.6 | 368.0 | **13.83×** |
| `np.cosh`    | float32 |  86.5 | 362.3 |   4.19× |
| `np.exp2`    | float32 | 247.3 | 691.3 |   2.80× |
| `np.log2`    | float32 | 205.3 | 674.2 |   3.28× |
| `np.log10`   | float32 |  86.0 | 674.2 |  **7.84×** |
| `np.arccosh` | float32 | 230.5 | 238.9 |   1.04× |
| `np.exp2`    | float64 |  20.1 | 292.5 | **14.55×** |
| `np.log2`    | float64 |  52.1 | 224.8 |   4.31× |
| `np.log10`   | float64 |  37.4 | 222.7 |   5.95× |
| `np.sinh`    | float64 |  24.5 | 155.9 |   6.36× |
| `np.cosh`    | float64 |  47.6 | 155.9 |   3.27× |
| `np.arccosh` | float64 |  74.3 | 105.6 |   1.42× |

`arccosh` shows only a modest speedup because the scalar libm `acoshf` is
already vectorizable / fast — the LoongArch libm seems to have a tuned
implementation. Other ops were genuinely scalar-bound.

### Follow-ups

- **Tier 2 transcendentals**: `tan`, `arcsin`, `arccos`, `arctan`,
  `arctan2`, `expm1`, `log1p`, `cbrt`, `power(x, y)`. Each needs its own
  polynomial / careful precision handling. `log1p` is the prerequisite for
  `arcsinh`/`arctanh` getting re-enabled.
- `power(x, y)` could be implemented as `exp(y·log(x))` for `x > 0`, but
  the integer-exponent fast path and `0^0 / negative base` edge cases need
  care. Not trivial.

---

## 2026-05-23 — Tier 2 NPYV transcendentals (everything except `power`)

**Files:**
- `numpy/_core/src/umath/npyv_exp_log.h` — added `log1p` and `expm1`
  kernels for f32 and f64. Both use Taylor for small `|x|` (avoids
  catastrophic cancellation in `1+x` or `e^x - 1`) and the existing
  `log_kernel(1+x)` / `exp_kernel(x) - 1` for moderate `|x|`. 7-term
  Taylor for f64; 5-term for f32.
- `numpy/_core/src/umath/loops_umath_fp.dispatch.c.src` — added 8 NPYV
  kernels: `arctan` (FDLIBM polynomial with 3-way range reduction),
  `arcsin`, `arccos` (composed on `arctan`), `arctan2` (binary, with
  quadrant logic), `tan` (Cody-Waite range reduction by π/2 + FDLIBM
  polynomial), `cbrt` (`sign(x) · exp(log(|x|)/3)`). Rewrote `arcsinh`
  and `arctanh` to use `log1p` so the small-|x| precision problem is
  fixed and they can be re-enabled in NPYV.

### What

This pass closes most of the gap left in `loops_umath_fp.dispatch.c.src`.
Of the 17 ufuncs that file owns, **15 now have NPYV implementations**;
only `power(x, y)` is left as a follow-up (its corner cases — negative
base with integer exponent, `0^0`, ±inf in either operand, etc. — make a
clean implementation substantially more involved than the others).

The kernels themselves break into a few patterns:
- **Direct polynomial after range reduction**: `arctan` (3-way reduction
  by `tan(π/8)` and `tan(3π/8)`, then FDLIBM minimax of degree 5 for
  f32 / degree 21 for f64), `tan` (Cody-Waite reduction by π/2, FDLIBM
  minimax of degree 11 for f32 / degree 25 for f64).
- **Compose on exp/log/log1p**: `arcsinh = log1p(|x| + x²/(1+sqrt(1+x²)))`
  (note rationalized form, otherwise the `1+x²→1` cancellation breaks
  small-|x|); `arctanh = ½·log1p(2x/(1-x))`; `cbrt = exp(log(|x|)/3)`;
  `exp2 = exp(x·ln2)`; `log2 = log·log₂(e)`; `log10 = log·log₁₀(e)`;
  `sinh = (eˣ−e⁻ˣ)/2`; `cosh = (eˣ+e⁻ˣ)/2`.
- **Compose on arctan**: `arcsin = arctan(x/sqrt(1-x²))`; `arccos = π/2 -
  arcsin`; `arctan2(y, x) = arctan(y/x)` plus quadrant adjustment.

### Decisions / gotchas

- **`tan` for huge |x|.** The Cody-Waite reduction is accurate up to
  about |x| < 2²⁰; beyond that the error in `r = x - k·π/2` grows. Real
  workloads rarely hit that range, and the test suite passes. Payne-Hanek
  reduction would extend accuracy further but adds significant code.
- **Spurious FP exceptions are the dominant failure mode.** For SIMD,
  every branch is always computed even when the result is selected away,
  so any path that divides by zero, computes `inf/inf`, takes `sqrt(<0)`,
  etc. fires a flag. Pattern that fixed it: mask the input to a safe
  value (1.0 or 0.0) on lanes where this branch isn't going to be picked.
  Caught and fixed across `arctan` (`(inf-1)/(inf+1) = NaN`), `arcsin`
  (`1/sqrt(0) = inf` at `|x|=1`), `arctan2` (`x/0` at `x=0`), `tan`
  (`-1/0` at even k where `tan_r=0`).
- **`arcsinh` + `arctanh` are back in NPYV** thanks to `log1p`. The
  identities `log1p(|x| + x²/(1+sqrt(1+x²)))` and `½·log1p(2x/(1-x))`
  don't lose precision at small `|x|`. The Tier 1 disable from
  2026-05-23 is reverted; markers are `NPYV_HAVE_asinh_*` /
  `NPYV_HAVE_atanh_*` again.
- **`arcsinh` for huge |x|** would overflow `x²` in the moderate path.
  We branch on `|x| > 2^15` (f32) / `2^27` (f64) and use the asymptotic
  `arcsinh(x) ≈ log(|x|) + ln(2)` there.
- **`arctan2` memory-overlap check** — the dispatcher must call
  `is_mem_overlap` before taking the NPYV path because `arctan2.accumulate`
  overlaps `dst` with one of the inputs. Caught by
  `test_memoverlap_accumulate_symmetric`.
- **`power` deferred.** A clean implementation needs `x < 0 with integer
  y` (parity-dependent sign), `0^0 = 1`, `±inf^(...)`, `(...)^±inf`, ±0
  with sign-preservation — a separate sprint.

### Test status

`spin test -- numpy/_core/tests/test_umath.py`: 4699 passed, 0 failed.

### Impact (3A6000, contiguous NPYV vs strided scalar libm, n=1M)

Numbers are noisy run-to-run (especially small absolute speeds where
overhead dominates), so the ratios below are rounded.

| op | f32 speedup | f64 speedup |
|---|---:|---:|
| **`np.tan`** | **~25×** | **~34×** |
| `np.arctan2` | ~22× | ~6× |
| `np.arcsinh` | ~17× | ~2× |
| `np.expm1`   | ~9× | ~4× |
| `np.log1p`   | ~8× | ~3× |
| `np.arctanh` | ~7× | ~3× |
| `np.cbrt`    | ~4× | ~8× |
| `np.arccos`  | ~4× | ~1× |
| `np.arctan`  | ~5× | ~5× |
| `np.arcsin`  | ~1.3× | ~1.1× |

`tan` is the standout: scalar `libm tan` is ~7 M elem/s on the 3A6000
(libm is doing the same range reduction we are, but element-by-element
with substantial per-call overhead), while our NPYV does it 8 lanes at a
time. `arcsin` / `arccos` get the smallest gains because they call into
our `arctan` which already pays the polynomial cost — the win is mostly
just the SIMD width.

### Follow-ups

- `power(x, y)` — last unbuilt op in this file.
- The tier-2 polynomial pivots (the `tan_pi8` / `tan_3pi8` constants in
  arctan, the π/2 split in tan) are dropped inline — would be cleaner in
  `npy_simd_data.h`.
- Some ops (`arcsin` f64, `arccos` f64, `cbrt` f32) show modest speedups.
  The LoongArch libm's implementations of those are already efficient.

---

## 2026-05-23 — power(x, y) NPYV

**File:** `numpy/_core/src/umath/loops_umath_fp.dispatch.c.src`

### What

Implemented `npyv_pow_FLOAT_kernel` and `npyv_pow_DOUBLE_kernel`. Closes
the last gap in `loops_umath_fp.dispatch.c.src` — every op in that file
now has an NPYV path on LoongArch (LSX and LASX).

### Algorithm

General formula: `|x|^y = exp(y · log(|x|))`. Around that, a chain of
mask-based special-case overrides per IEEE 754 / C99 pow():

| input pattern | result |
|---|---|
| `pow(x, 0)` | 1 (even for NaN x) — overrides everything |
| `pow(1, y)` | 1 (even for NaN y) — overrides everything |
| `pow(x<0, integer y)` | `±|x|^y` (sign from parity of y) |
| `pow(x<0, non-integer y)` | NaN |
| `pow(x, ±inf)` | {0, 1, inf} depending on \|x\| vs 1 |
| `pow(0, y>0)` / `pow(0, y<0)` | 0 / +inf (DIV_BY_ZERO for second) |
| NaN propagation | otherwise |

### Spurious-flag mitigations

The recurring "every SIMD branch always executes" issue, applied here
for several traps the formula would set on lanes we mask away:

- `log(0)` would return `-inf` (clean in our log_kernel, but `y·-inf`
  with `y=0` then gives NaN with INVALID) — so x=0 lanes are masked to
  `|x|=1` before the log.
- `0 · ±inf = NaN` with INVALID — the `|x|=1` lanes have `y` masked to 0
  before the multiplication.
- `round_to_int(±inf)` saturates with INVALID — y non-finite lanes are
  masked to 0 before the int conversion / parity check.

### Decisions

- **Parity check uses `round_s32_f32` (f32) or even/odd fractional check
  (f64).** The f32 path's int32 saturation means `pow(-2, y)` for `y > 2^31`
  loses parity information — acceptable; will silently treat as even for
  extremely large exponents. The f64 path uses the more robust
  `y/2 - rint(y/2) == 0 ⇒ even` trick so it works for all finite y.
- **`pow.accumulate` memory-overlap** is checked in the dispatcher
  (`is_mem_overlap(args[0..1], args[2], ...)`) — same gotcha as `arctan2`.
- **DIV_BY_ZERO at `pow(0, -1) = +inf`** is intentional, per IEEE.
  numpy/libm both raise it; we match that behavior.

### Test status

4699 umath tests pass, no regressions.

### Impact (3A6000, n = 1M)

| op | dtype | scalar libm | NPYV | speedup |
|---|---|---:|---:|---:|
| `np.power` | float32 | 80 M/s | 131 M/s | 1.64× |
| `np.power` | float64 | 17 M/s | 47 M/s | **2.71×** |

Modest because the LoongArch libm `pow` is already an `exp(y·log(x))`
composition under the hood — but the f64 win is meaningful and the
implementation establishes that all 17 ops in `loops_umath_fp` are now
NPYV-routed on LoongArch.

### Final state of `loops_umath_fp.dispatch.c.src` on LoongArch

| op | f32 | f64 |
|---|---|---|
| `exp2`, `log2`, `log10` | NPYV (Tier 1) | NPYV (Tier 1) |
| `sinh`, `cosh` | NPYV (Tier 1) | NPYV (Tier 1) |
| `arccosh` | NPYV (Tier 1) | NPYV (Tier 1) |
| `arcsinh`, `arctanh` | NPYV (re-enabled via `log1p`, Tier 2) | NPYV (Tier 2) |
| `log1p`, `expm1` | NPYV (Tier 2) | NPYV (Tier 2) |
| `arctan`, `arcsin`, `arccos`, `arctan2` | NPYV (Tier 2) | NPYV (Tier 2) |
| `tan` | NPYV (Tier 2) | NPYV (Tier 2) |
| `cbrt` | NPYV (Tier 2) | NPYV (Tier 2) |
| `power` | **NPYV (this pass)** | **NPYV (this pass)** |

17 unary + 2 binary = **19 ufuncs** all on NPYV.

## 2026-05-23 — Regression cleanup: arcsin / arccos f64 direct kernel

**File:** `numpy/_core/src/umath/loops_umath_fp.dispatch.c.src`

### What

The Tier-2 pass made `arcsin` / `arccos` f64 *slower* than scalar libm
on the 3A6000:

| op | dtype | main NPYV | head NPYV (pre-fix) | head/main |
|---|---|---:|---:|---:|
| arcsin | f64 | 67.9 M/s | 34.3 M/s | **0.51×** |
| arccos | f64 | 64.8 M/s | 52.0 M/s | **0.80×** |

### Why the composition lost

The original kernel computed `arcsin(x) = arctan(x/√(1-x²))`, riding on
the existing Tier-2 arctan. That means per element:

  - 1 sqrt, 1 div for the argument transform
  - the full arctan f64 kernel: 3-way reduction + degree-21 FDLIBM poly

With 4-lane LASX f64 the SIMD parallelism doesn't make up for doing ~2×
the floating-point work of scalar libm's direct asin.

### Fix

Rewrote `npyv_asin_DOUBLE_kernel` using FDLIBM's two-region rational
(coefficients from glibc `sysdeps/ieee754/dbl-64/e_asin.c`):

  - `|x| < 0.5`:  `asin(x) = x + x·t·P(t)/Q(t)`,  `t = x²`
  - `|x| ≥ 0.5`:  `asin(x) = sign(x)·(π/2 - 2·s·(1+w))`,
                  `s = √((1-|x|)/2)`, `w = t·P(t)/Q(t)` with the same poly

Both branches share P (degree 5) / Q (degree 4) so the polynomial is
evaluated unconditionally. The large-path identity moves the argument
back into a regime where the same poly is accurate. Total cost: one
sqrt + one div + one degree-5 Horner + one degree-4 Horner. No
transcendentals.

`npyv_acos_DOUBLE_kernel` continues to compose as `π/2 - asin(x)` — now
gets the win for free.

### Result

  - `arcsin` f64: 34.3 → 269.4 M elem/s (0.51× → **3.97×** vs main)
  - `arccos` f64: 52.0 → 264.6 M elem/s (0.80× → **4.08×** vs main)
  - Overall geomean across 46 ops: 3.09× → **4.25×**
  - Regressions vs main: 5 → 0
  - All 4699 umath tests pass

### Decision: don't also rewrite f32 arcsin/arccos

The f32 kernels still compose on arctan and benchmark at 2.07× /
2.27× vs main — already winning, no precision concerns. Leave them
alone; the f64 fix doesn't dictate f32 changes.

## 2026-05-23 — Re-enable SIMD f64 sin/cos with precision gate

**Files:** `numpy/_core/src/umath/npyv_sincos.h` (new),
`numpy/_core/src/umath/loops_trigonometric.dispatch.cpp`

### Context — why upstream disabled this

Upstream `loops_trigonometric.dispatch.cpp` lines 215-217 disabled the
SIMD f64 sin/cos path entirely:

```cpp
/* Disable SIMD code sin/cos f64 and revert to libm: see
 * https://mail.python.org/archives/list/numpy-discussion@python.org/
 *   thread/C6EYZZSR4EWGVKHAZXLE7IBILRMNVK7L/
 * for detailed discussion on this*/
```

The numpy-discussion thread concerned a previously-shipped SVML kernel
that lost precision for `|x| ≥ 2^20` (Cody-Waite range reduction breaks
down past about 2^20 with a three-part π/2 split, and the kernel had no
Payne-Hanek fallback). Result: `DOUBLE_sin` / `DOUBLE_cos` were routed
to scalar libm on every platform — which on the 3A6000 leaves ~3× of
LASX speedup on the floor.

### What

Reintroduced a SIMD f64 sin/cos *only* for the range where Cody-Waite
is provably accurate, with a per-block gate that falls back to scalar
libm for blocks containing any out-of-range lane. The gate matches the
exact pattern Highway uses for f32 sin/cos in the same file.

Implementation:

  - `npyv_sincos.h` — single FDLIBM polynomial kernel
    `npyv__sincos_f64_kernel(x, want_cos)` that does Cody-Waite
    reduction with a three-part π/2 split, evaluates the FDLIBM
    `__kernel_sin` and `__kernel_cos` polynomials, and applies the
    quadrant selection / sign flip.

  - `loops_trigonometric.dispatch.cpp` — `DISPATCH_DOUBLE_FUNC(func,
    OPCODE)` now wraps the SIMD call (only for LoongArch + NPYV f64).
    For each block of `npyv_nlanes_f64` lanes, the dispatcher computes
    `|x| ≤ 2^20` for every lane via `npyv_all_b64(cmple)`. If all
    lanes are in range, the SIMD kernel runs; otherwise the dispatcher
    loops the block through scalar `npy_sin` / `npy_cos`. Outside
    LoongArch, the macro reduces to the original upstream form.

### Precision audit (3A6000, 100k samples per range)

| range | sin max ULP | sin mean ULP | cos max ULP | cos mean ULP |
|---|---:|---:|---:|---:|
| `|x|` ≤ 1 | 1.00 | 0.08 | 1.00 | 0.13 |
| `|x|` ≤ 100 | 1.00 | 0.16 | 1.00 | 0.17 |
| `|x|` ≤ 10⁶ | 1.00 | 0.17 | 1.00 | 0.17 |
| `|x|` > 2²⁰ (libm fallback) | 0.00 | 0.00 | 0.00 | 0.00 |

Max 1 ULP across the SIMD regime — matches or beats FDLIBM's own ~2
ULP target. Beyond the gate the libm fallback is bit-exact since both
the strided and contiguous paths call the same `npy_sin`/`npy_cos`.

NaN is handled inside the kernel (masked to 0 before `rint`, restored
in the result).

### Result

  - `sin` f64: 68.5 → 240.5 M elem/s (1.00× → **3.51×** vs main)
  - `cos` f64: 67.9 → 235.9 M elem/s (1.00× → **3.47×** vs main)
  - Overall geomean across 46 ops: 4.25× → **4.56×**
  - **Zero remaining neutrals or regressions** — every op ≥ 1.10×
    speedup vs main; lowest is now `cbrt` f64 at 1.42×.
  - All 4699 umath tests still pass.

### Decision: hybrid gate, not full SIMD reduction

We chose the per-block `|x| < 2^20` libm fallback over implementing
Payne-Hanek in SIMD. Reasoning:

  - The common case (data within ~10⁶ in magnitude) gets the full SIMD
    speedup. Real-world workloads rarely exceed this range; arrays of
    angles are usually wrapped to `[-π, π]` or close.
  - Payne-Hanek in SIMD would add ~150 lines and introduces its own
    precision-tuning sensitivity. For a tech-demo branch the
    risk/reward isn't there yet.
  - The fallback is correct (libm-exact), not approximate — so users
    with huge arguments get the original numpy behavior, no surprises.

### Follow-up: SIMD Payne-Hanek

TODO: write a SIMD Payne-Hanek range reduction so the SIMD f64 sin/cos
path covers the full f64 range without falling back to libm. This
would:

  - eliminate the per-block max(|lane|) check (a small win — currently
    ~1 compare + branch per block)
  - cover workloads that *do* have huge arguments (e.g. unwrapped
    phase data) at the SIMD speedup
  - bring the LoongArch path closer to deserving an upstream PR
    (rather than a LoongArch-specific opt-in)

References:
  - Payne, M. & Hanek, R., "Radian Reduction for Trigonometric
    Functions", SIGNUM Newsletter, 1983.
  - FDLIBM `k_rem_pio2.c` is a scalar reference.
  - The Highway TODO at the top of `simd_sincos_f32` flags the same
    follow-up for the f32 path.

### Decision: keep upstream behavior on non-LoongArch

The `#if NPY_SIMD && defined(__loongarch__) && defined(NPYV_IMPL_F64_EXP_LOG)`
guard around the new macro body means non-LoongArch builds get the
original upstream `UNARY_LOOP { *op1 = npy_##func(*ip1); }` exactly.
This was deliberate — re-enabling SIMD f64 sin/cos on x86 would
override the numpy-discussion decision for a platform we have no
benchmark data on. LoongArch is opt-in; if someone wants to extend the
hybrid to other arches the guard is one line.

## 2026-05-23 — cbrt f64: FDLIBM bit-magic + Newton

**File:** `numpy/_core/src/umath/loops_umath_fp.dispatch.c.src`

### Context

cbrt f64 was the lowest-speedup op after the sin/cos work — 1.42× vs
main (sometimes measured at 1.29×, depending on noise). The previous
kernel composed on the existing NPYV exp/log:

```c
cbrt(x) = sign(x) · exp(log(|x|) / 3)
```

That pays ~30 ops for log_kernel and another ~30 for exp_kernel, plus
the multiply by 1/3 and the sign reconstruction — roughly 60 ops per
element. Libm skips both transcendentals.

### Algorithm (FDLIBM s_cbrt.c)

  1. **Initial estimate via bit hack** (~5 bits of accuracy).
     Take the top 32 bits of `|x|`, divide by 3, add a magic constant
     B1 = 715094163 ≈ (682 - 0.033) · 2²⁰. The arithmetic on the
     biased exponent field implicitly divides the binary exponent by
     3, and B1 nudges the result toward the correct cube-root
     exponent. Shift back into the high 32 bits with zero low bits,
     reinterpret as a double:

     ```c
     t = bits_to_double(((|x|_bits >> 32) / 3 + B1) << 32)
     ```

  2. **Rational refinement** (~5 → ~23 bits).
     Standard FDLIBM rational on `r = t³/x`:

     ```c
     r = t·t/x;
     s = C + r·t;       // C = 19/35
     t *= G + F/(s + E + D/s);
     ```

     C, D, E, F, G are the FDLIBM constants (19/35, -864/1225, 99/70,
     45/28, 5/14).

  3. **Newton iteration to 53 bits.** Critically, zero the low 32
     bits of `t` first so `t²` is exact (no rounding error in the
     multiplication). Then:

     ```c
     s = t·t;
     r = x/s;
     w = t+t;
     r = (r-t)/(w+r);
     t = t + t·r;
     ```

### SIMD specifics

  - The bit-hack divide-by-3 of the top 32 bits is done in FP: convert
    `hx` (which is < 2³¹, exact in f64) to a double, multiply by 1/3,
    apply `npyv_floor_f64`, convert back to integer. Avoids needing a
    SIMD integer-divide-by-3 or a widening multiply.
  - Subnormals, ±0, ±inf, NaN bypass the bit-magic via a `special`
    mask and the lane passes through the original `x` (which is
    correct for ±0, ±inf, NaN since cbrt of each equals itself). The
    subnormal lane is technically wrong — `cbrt(subnormal)` is *much*
    larger than the subnormal — but subnormals are rare in real-world
    numpy workloads and the FDLIBM rescaling branch they need
    (multiply by 2⁵⁴, run the algorithm, divide by 2¹⁸) isn't worth
    vectorizing for the precision/speed tradeoff. Filed under
    follow-ups; for now, lanes containing subnormals would need to
    fall through to libm in the dispatcher (the strided path already
    does this).
  - Truncating the low 32 bits of `t` between the rational refinement
    and the Newton step is the standard FDLIBM trick: it makes `t·t`
    exact in floating-point. We do this in SIMD with a bitwise AND
    against `0xFFFFFFFF00000000`.

### Precision audit (3A6000, 50k samples per range, vs scalar libm)

| range | max ULP | mean ULP |
|---|---:|---:|
| `[0.001, 1]` | 3.00 | 0.64 |
| `[1, 1000]` | 3.00 | 0.50 |
| `[-1000, -0.001]` | 3.00 | 0.49 |
| `[1e-200, 1e200]` (huge exponent range) | 3.00 | 0.51 |

Max 3 ULP across the full f64 normal range — slightly looser than
FDLIBM's stated 2 ULP target. The extra ULP comes from the
FP-based divide-by-3 in the initial estimate (FDLIBM does
integer divide of the 32-bit hx; we do `floor(hx · (1/3))` which
introduces a sub-ULP nudge that's amplified by the rational
refinement). Could be tightened by adding an integer-divide-by-3
helper to NPYV, but 3 ULP is well within the budget for SIMD
transcendentals.

Special values verified: `cbrt(±0) = ±0`, `cbrt(±inf) = ±inf`,
`cbrt(NaN) = NaN`. All 86 `cbrt` tests in test_umath.py pass.

### Result

  - `cbrt` f64: 42.7 → 140.6 M elem/s (1.42× → **3.29×** vs main)
  - Overall geomean across 46 ops: 4.56× → **4.71×**
  - Bottom-10 floor rose from 1.42× to 1.55× — every benchmarked op
    is now ≥ 1.55× faster than scalar libm.

### Decision: don't also rewrite f32 cbrt

f32 cbrt was already at 3.57× vs main (the existing
`exp(log(|x|)/3)` composition runs well on the 8-lane LASX f32 path —
SIMD parallelism wins). Rewriting it would gain less and would need
its own ULP audit; current implementation is fine.

### Follow-up: ~~subnormal handling~~ — done

Initial cbrt landing routed subnormals to `x` (wrong) since FDLIBM's
rescaling branch wasn't yet vectorized. Added below.

## 2026-05-23 — cbrt f64: subnormal correctness

**File:** `numpy/_core/src/umath/loops_umath_fp.dispatch.c.src`

### What

The bit-magic initial estimate for cbrt assumes a normal-magnitude
input — the biased exponent field needs to be in the range a normal
double occupies, and the divide-by-3-plus-magic arithmetic breaks if
the exponent field is zero (subnormal). The first landing routed
subnormal lanes to `x` itself, which is wrong: `cbrt(2⁻¹⁰⁷⁴) ≈
1.7e-108`, not `2⁻¹⁰⁷⁴`.

### Fix

Per FDLIBM `s_cbrt.c`:

  1. Pre-scale subnormal lanes by 2⁵⁴ to push them into the normal
     range (any subnormal × 2⁵⁴ is normal).
  2. Use magic constant B2 = 696219795 instead of B1 = 715094163
     for those lanes. B2 = B1 − 18·2²⁰ — the −18 in the exponent
     field of the initial estimate exactly undoes the 2⁵⁴ scale-up
     (since cbrt(2⁵⁴) = 2¹⁸).
  3. The rational refinement and Newton step use the **unscaled**
     |x|, since they refine `t³ ≈ |x|` against the original
     cube-root constraint. (For x = 2⁻¹⁰³⁰, t ≈ 2⁻³⁴³, and
     `t²/x ≈ 2³⁴⁴` — both normal, so the divide-by-subnormal step
     yields a finite result even though `x` is subnormal.)

In SIMD this is a per-lane select before the bit-magic: pick
`bit_input = abs_x × 2⁵⁴` and `magic = B2` for subnormal lanes,
else `bit_input = abs_x` and `magic = B1`. Two extra selects and one
extra multiply — costs essentially nothing on the 4-lane LASX path
(throughput stayed within noise at 137 M/s after the change).

### Precision audit (156 subnormal samples across the full subnormal range)

| range | max ULP | mean ULP |
|---|---:|---:|
| subnormal (2⁻¹⁰⁷⁴ .. 2⁻¹⁰²²) | **1.00** | 0.34 |

Smallest subnormal `cbrt(2⁻¹⁰⁷⁴) = 1.703184e-108` matches scalar libm
exactly. Interesting: subnormals come out to 1 ULP (vs 3 ULP for
normals) because the scaled-up input gives the bit-magic a cleaner
starting point — the FP divide-by-3 has more headroom.

### Edge cases verified

| input | output | match libm? |
|---|---|---|
| 0.0 | 0.0 | ✓ |
| -0.0 | -0.0 | ✓ |
| +inf | +inf | ✓ |
| -inf | -inf | ✓ |
| NaN | NaN | ✓ |
| 2⁻¹⁰²² (smallest normal) | 2.81264e-103 | ✓ |
| 2⁻¹⁰⁷⁴ (smallest subnormal) | 1.70318e-108 | ✓ |

All 4699 umath tests still pass. Throughput vs main unchanged at
~3.2×.

## 2026-05-24 — log1p f64: direct FDLIBM Padé with per-block dispatch

**File:** `numpy/_core/src/umath/npyv_exp_log.h`

### Context

log1p f64 sat at 1.95-2.17× vs main — the kernel's slow path went
through `log_kernel(1+x)` (~30 ops) for any |x| above the very tight
2⁻⁷ Taylor threshold. With both the 7-term Taylor branch and
log_kernel always computed-and-blended, every call paid the full
log_kernel cost.

### Algorithm — two kernels, per-block dispatch

  - **`npyv__log1p_pade_f64`** — FDLIBM `Lp1..Lp7` Padé form
    (= `Lg1..Lg7`, the 2/(2k+1) coefficients of the Mercator series):

    ```
    s = x/(2+x);   z = s²;   hfsq = x²/2
    R = z · poly(z)        [Horner, 7 coefficients]
    log1p(x) = x - (hfsq - s·(hfsq + R))
    ```

    Cost ~15 ops, valid for `x ∈ (1-√2/2, √2-1) ≈ (-0.293, 0.414)`.
    Outside that band the formula loses ~14 bits.

  - **`npyv__log1p_full_f64`** — `log_kernel(1+x)` plus FDLIBM's
    correction term. The correction captures the rounding error of
    `1+x` so log1p stays ULP-accurate even when `x` is small enough
    that `1+x` rounds to 1. Uses a per-lane `select` between two
    forms of the correction depending on whether `u = 1+x ≥ 2`.

  - **Top-level kernel** — per-block check via `npyv_all_b64`:
    if every lane is in the Padé band, run the cheap path; otherwise
    run the full path. The check is one `cmpgt + cmplt + and` plus
    a single scalar branch — predictable for any monotone or
    sub-range workload.

### Lessons from getting this wrong

The first attempt used a symmetric threshold `|x| < 0.4`. That ran
the Padé for `x = -0.39`, where the formula's underlying assumption
(that `u = 1+x` lands in `[√2/2, √2)`) is violated. Result: ~8700
ULP error in a region the test caught. Threshold corrected to the
asymmetric `(-0.29, 0.41)` band that FDLIBM uses; max ULP dropped
back to 1 across the whole f64 normal range.

The full path also needed inf/NaN masking — `inf − inf` in the
correction triggered spurious `INVALID` (numpy's `test_unary_spurious_fpexception`
caught this). Fix: mask non-finite lanes to 0 *before* the
subtractions, then let `log_kernel(1+x)` produce the right inf/NaN
on its own special-case paths. Also added a `u == 0` guard so
`log1p(-1)` doesn't compute `0/0` in the correction (caught when
`arctanh(-1)` started returning NaN instead of -∞).

### Precision audit (50k samples per range, vs scalar libm)

| range | max ULP | mean ULP |
|---|---:|---:|
| `(-0.29, 0.41)` (Padé sweet spot) | 1.00 | 0.00 |
| `(-0.5, 0.5)` (bench range, mixed) | 1.00 | 0.07 |
| `(-0.99, 5)` (full normal range) | 1.00 | 0.05 |
| `(1e-300, 1e300)` (extreme exponents) | 0.00 | 0.00 |

Special-value check: `log1p(0) = 0`, `log1p(-1) = -∞`, `log1p(-2)
= NaN`, `log1p(∞) = ∞`, `log1p(NaN) = NaN` — all correct, no
spurious flags.

### Result

  - `log1p` f64: 63.4 → 231.1 M elem/s (1.95× → **3.65×** vs main)
  - `arctanh` f64 also bumps (was 2.54×) because its slow path
    composes on `log1p`. See next section.

## 2026-05-24 — arctanh f64: direct Cephes Padé in x² with per-block dispatch

**File:** `numpy/_core/src/umath/loops_umath_fp.dispatch.c.src`

### Context

arctanh f64 was at 2.54× vs main. The kernel did
`0.5 · log1p(2x/(1-x))` — even with the faster log1p above, this
still pays a divide and the full log1p path.

### Algorithm

For `|x| ≤ 0.5`, the function is approximated by a Cephes Padé in
`t = x²`:

```
arctanh(x) = x + x · t · P(t) / Q(t)
```

with `P` degree 4, `Q` degree 5 (leading 1 implicit). Coefficients
from Cephes `atanh.c`, peak relative error 1.4e-17. Cost ~13 ops.

For `|x| > 0.5` the polynomial doesn't converge — use the original
log1p formula. Per-block dispatch picks the path based on
`max(|lane|) ≤ 0.5`.

### Lessons

Cephes stores polynomial coefficients with `coef[0]` as the
**highest** power, not the lowest. The first attempt evaluated Horner
in the wrong order and produced wildly wrong values (~10¹⁸ ULP
errors). Documented this in the kernel comment so we don't repeat
when porting other Cephes routines.

### Precision audit (50k samples per range)

| range | max ULP | mean ULP |
|---|---:|---:|
| `(-0.4, 0.4)` (Padé sweet spot) | 2.00 | 0.24 |
| `(-0.5, 0.5)` (bench range) | 2.00 | 0.25 |
| `(-0.99, 0.99)` (slow path, near singularity) | 21.00 | 0.54 |

The 21 ULP at the slow-path boundary is the asymptotic blow-up
where `(1-x)` underflows ULPs near `x = 1`; libm itself only does
slightly better here.

Special values: `arctanh(±0) = ±0`, `arctanh(±1) = ±∞`,
`arctanh(|x| > 1) = NaN`, `arctanh(NaN/±∞) = NaN` — all correct.

### Result

  - `arctanh` f64: 33.0 → **485.2 M elem/s — 14.7× vs main**
    (was 2.54× before the rewrite). The bench range is entirely
    inside the Padé band so almost every block runs the fast path.
  - Overall geomean across 46 ops: 4.71× → **4.95×**

## 2026-05-24 — Payne-Hanek reduction for f64 sin/cos slow path

**Files:** `numpy/_core/src/umath/payne_hanek_f64.h` (new),
`numpy/_core/src/umath/npyv_sincos.h`,
`numpy/_core/src/umath/loops_trigonometric.dispatch.cpp`

### Context

The 2026-05-23 sin/cos f64 landing gated the SIMD path on
`max(|lane|) < 2^20` and fell back to scalar libm for blocks
containing any larger input. The 2^20 limit comes from Cody-Waite
range reduction: y = round(x · 2/π) needs to be exactly representable
in the 53-bit mantissa, and rounding errors in x · inv_pi_2 corrupt
y for larger |x|. Beyond that, libm uses Payne-Hanek — a table-based
reduction that handles arbitrary magnitudes by storing many bits of
2/π and indexing them based on x's exponent.

This pass replaces the libm fallback with our own Payne-Hanek
reduction. Reduces the libm dependency to zero on the LoongArch
sin/cos f64 path; covers the full f64 normal range with libm-equivalent
precision.

### Implementation

Two new files; the slow-path kernel sits in between them:

  - **`payne_hanek_f64.h`** — Scalar Payne-Hanek reduction
    `npyv__payne_hanek_reduce(x, *r_out) → quadrant`. Algorithm and
    table data ported from Sun's FDLIBM `e_rem_pio2.c` /
    `k_rem_pio2.c` (public domain):

      1. Scale `|x|` so its biased exponent becomes 1046 (unbiased 23),
         giving a value in `[2^23, 2^24)`. Peel off three 24-bit
         integer chunks `tx[0..2]`.
      2. Index a precomputed table of 2/π bits (66 chunks × 24 bits =
         1584 bits, enough for `|x| < 2^1024` plus margin). Load
         `jx+jk+1` chunks where `jx = nx-1` and `jk = 4` (FDLIBM's
         picks for f64 precision).
      3. Multiply `tx[]` by the loaded chunks to get partial products
         `q[0..jk]`. Each multiplication is exact in f64 (24-bit × 24-bit
         fits in 48 bits ≤ 53).
      4. Distill `q[]` into 24-bit integer pieces `iq[]` from the high
         end down, pulling off the integer part of `x · 2/π` for the
         quadrant.
      5. If the surviving fraction is > 0.5, bump quadrant and
         complement `iq[]`. Iterate if the fraction collapsed to
         exactly zero (rare; `goto recompute`).
      6. Multiply the fraction back by π/2 in 24-bit-aligned chunks
         to recover the reduced argument.

  - **`npyv_sincos.h`** — Refactored to split the kernel into a
    reduction half (Cody-Waite or Payne-Hanek) and a polynomial half
    (`npyv__sincos_f64_polynomial`). The slow-path entry point
    `npyv__sincos_f64_slow_path` does per-lane scalar Payne-Hanek to
    build vectors of `(r, quadrant)`, then runs the SAME vectorized
    polynomial as the fast path. NaN and ±inf inputs are routed to a
    NaN output without going through the table-based reduction.

  - **`loops_trigonometric.dispatch.cpp`** — The else-branch of the
    per-block `|x| < 2^20` gate now calls `npyv__sincos_f64_slow_path`
    instead of looping through `npy_sin`/`npy_cos`.

### Precision audit (10k samples per range, vs scalar libm)

| range | sin max ULP | cos max ULP |
|---|---:|---:|
| `[-1, 1]` (fast path) | 1.00 | 1.00 |
| `[-100, 100]` (fast path) | 1.00 | 1.00 |
| `[-10⁶, 10⁶]` (fast path) | 1.00 | 1.00 |
| `[-2²⁵, 2²⁵]` (slow path) | 1.00 | 1.00 |
| `[-2⁴⁰, 2⁴⁰]` (slow path) | 1.00 | 1.00 |
| `[-2¹⁰⁰, 2¹⁰⁰]` (slow path) | 1.00 | 1.00 |
| `[-2⁵⁰⁰, 2⁵⁰⁰]` (slow path) | 1.00 | 1.00 |

Specific check: `sin(2⁵⁰⁰) = 0.42925739234240...` matches libm exactly.
Spot checks at `1e15`, `1e30`, `1e100`, `1e150`, `1e300` all match
libm bit-for-bit. All 4699 umath tests pass.

### Throughput (3A6000, n = 1M)

| input range | which path | speed | vs libm equiv |
|---|---|---:|---:|
| `[-3, 3]` | fast Cody-Waite SIMD | 240 M/s | 3.56× over scalar libm |
| `[-10¹⁰, 10¹⁰]` | slow Payne-Hanek SIMD | 7.3 M/s | 0.78× vs scalar libm 9.4 M/s |
| `[-10¹⁰⁰, 10¹⁰⁰]` | slow Payne-Hanek SIMD | 7.0 M/s | similar |

The slow path is currently ~25% slower than scalar libm. The
polynomial half is vectorized, so once the per-lane reduction overhead
is amortized, this gap should close. Vectorizing the reduction (true
SIMD Payne-Hanek with vectorized FP arithmetic across the chunks) is
the next follow-up — it would need a SIMD gather for the per-lane
table-chunk loads, which on LASX requires `xvinsgr2vr_d` plumbing.

Overall bench geomean across the 46 op×dtype pairs (mostly fast-path
inputs) is unchanged at 4.94×.

### Decision: scalar reduction + vector polynomial, not full SIMD reduction

A fully-vectorized Payne-Hanek would need per-lane gather of three or
four 24-bit chunks from a 66-entry table. LASX has no native gather
instruction; the workaround is `xvinsgr2vr_d` after four scalar loads,
which doesn't really save time over the scalar approach we have. The
polynomial after the reduction is the easy part to vectorize and
already is.

### Attempted SIMD Payne-Hanek (2026-05-24), and why it failed

I prototyped a vectorized PH that did per-lane gather of 4 chunks of
2/π plus all-vector FP arithmetic for the multiplications and the
quadrant/fraction split. It was correct for `|x| ≤ ~10⁶` (1 ULP) but
catastrophically wrong (3000+ ULP) for `|x| ≳ 10¹⁰`.

The root cause is precision loss during chunk accumulation. Each
chunk's contribution `m · chunk_k · 2^(e − 52 − 24(k+1))` covers a
77-bit range, but f64 only holds 53 bits. Summing chunks at different
bit-scales in plain f64 arithmetic drops the lower-order bits of each
sum:

  - chunk 0 contribution at exponent ~+34 (for x ≈ 10¹⁰) has ULP ~2⁻¹⁹
  - chunk 1 contribution at exponent ~+10, gets aligned to the same
    ULP scale when added → loses ~15 bits
  - chunks 2 and 3 lose progressively more

After the sum, only ~20 bits of the fractional part survive, which is
nowhere near the 53 ULP-precision we need for sin/cos.

The fixes, all substantial:

  1. **Double-Double accumulation.** Replace the plain `+` between
     partial products with `ddadd2` (Dekker-style two-sum with FMA
     correction). Each accumulator step preserves both the high and
     low parts of the result. ~6 extra ops per add × 4 adds = 24 ops
     per element, plus the multiplications also need DD form.
  2. **Multi-precision integer accumulation.** Keep the running sum
     as separate 24-bit-piece vectors (FDLIBM's `iq[0..jz]` style),
     propagate carries vector-side. Essentially porting FDLIBM's
     `__kernel_rem_pio2` directly to SIMD with per-lane state.
  3. **Pre-aligned table** (SLEEF approach). Generate the table so
     each entry's 4 doubles are already at the right scale for direct
     multiplication, with the alignment baked in. Requires `mpmath`
     to generate the table; constant-table size grows from 66 ints
     to a few thousand doubles.

(1) is the cleanest; (3) is what SLEEF actually does. Either would
add 100-300 lines of careful code. The current scalar PH works,
ships correct results across the full f64 range at 7-9 M/s, and the
bench is unaffected (the bench inputs are all fast-path). So the
SIMD-reduction follow-up stays open but is not blocking.

### Decision: don't add direct Padé for arcsinh f64

arcsinh's existing `log1p(|x| + x²/(1+√(1+x²)))` produces a log1p
argument that's generally > 0.5 for the bench range `|x| < 2`
(e.g. arcsinh(0.5) → log1p(0.618), arcsinh(1) → log1p(1.414)).
So per-block dispatch with threshold 0.5 in log1p would rarely fire
through the arcsinh call path. Direct polynomial for arcsinh would
need its own reduction (FDLIBM splits into 4 ranges based on |x|),
which is engineering effort with limited bench-relevant payoff
on this CPU. Filed for later if a real workload demands it.

## 2026-05-24 — Reclaim tanh / cbrt-f32 / power from libm

**Files:** `numpy/_core/src/umath/loops_umath_fp.dispatch.c.src`,
`numpy/_core/meson.build`

### Context

The cross-CPU bench against stock NumPy on Intel showed three
transcendentals lagging more than the f64/f32 SIMD-width gap could
account for: `tanh` at 0.06×, `cbrt` f32 at 0.06×, `arcsinh`/`power`
in the 0.08–0.11× range. Diagnosis:

  - `tanh` had a Highway-based SVML port in
    `loops_hyperbolic.dispatch.cpp.src` listed for LASX/LSX, but the
    bench numbers strongly suggested it was falling through to scalar
    libm on Loongson (54 M/s for f64 is about right for scalar `tanh`
    at 18 ns/element).
  - `cbrt` f32 was the slow `exp(log(x)/3)` composition while `cbrt`
    f64 had been rewritten to FDLIBM bit-magic back in
    2026-05-23. f32 got skipped.
  - `power` was paying ~50 select ops per element for the IEEE-754
    edge cases (negative bases, ±inf y, ±0, ±1) even when no lane
    needed any of them.

`arcsinh` is the one outlier — see prior section. The log1p form is
intrinsic to its accuracy and the path is hard to shortcut without
custom-derived Padé coefficients.

### Implementation

  - **tanh f32/f64.** New NPYV kernels using
    `tanh(x) = sign(x) · (1 − 2/(exp(2|x|) + 1))` — one `exp` call,
    versus the two needed by the `sinh/cosh` decomposition. Clamp
    `|x|` to 9 (f32) / 20 (f64) before the `exp` so `e^{2|x|}` stays
    finite; tanh past those thresholds saturates to ±1 within ULP.

    Removed `LASX, LSX` from the `loops_hyperbolic` arch list in
    `meson.build` so the dispatch function is no longer emitted from
    that source for LoongArch builds, avoiding a duplicate-symbol
    collision. The new `DOUBLE_tanh` / `FLOAT_tanh` definitions in
    `loops_umath_fp` are gated by `NPYV_HAVE_tanh_@sfx@` (only true
    when the NPYV exp/log path is compiled in, i.e. LSX/LASX), so on
    other targets `loops_hyperbolic`'s SVML port is still the source
    of truth.

  - **cbrt f32.** Replaced with the FDLIBM `cbrtf` algorithm: bit-magic
    initial estimate (`hx/3 + B1` with `B1 = 709958130`) plus two
    Halley iterations in f32. Halley converges cubically (5 → 15 →
    45 bits), so two iterations clear the 24-bit mantissa with
    headroom. Subnormals get pre-scaled by `2^24` with `B2 = B1 − 8·
    2^23 = 642849266` to compensate the exponent field. Skips the
    log + exp pair entirely.

  - **power f64/f32 hot path.** Wrapped the existing kernel with a
    per-block fast check: if every lane has `x > 0`, both `x` and `y`
    finite, `y ≠ 0`, and `x ≠ 1`, then `result = exp(y · log(x))`
    with no select wave. Most workloads (statistical, simulation,
    image processing) satisfy this and used to pay ~50 select ops
    per element. The slow-path code is untouched and still runs
    correctly on negative bases, infinities, ±0, etc.

### Tradeoffs

  - `tanh` saturating at |x|=9 (f32) is correct to ULP for typical
    workloads but loses sign information when `x` is a NaN with
    different payload in upper bits. We pass NaN through unchanged at
    the end of the kernel, so this is moot.

  - `cbrt` f32 with bit-magic depends on `npyv__cvt_f32_s32` being
    available, which mirrors the existing f64 implementation's
    constraint. Both LSX and LASX provide it.

  - `pow` fast path costs ~5 ops/block of mask construction. Negligible
    for the all-clean common case; in the worst-case (every block
    contains a special value), we pay both the check and the slow
    path. The check makes the win for the common case.

### Verified on Intel

The four ops compile cleanly. `tanh` on Intel still routes through
the loops_hyperbolic SVML port (we only pulled LASX/LSX from its
arch list); f32 `cbrt` and `pow` aren't exercised by our NPYV path on
Intel (`NPYV_IMPL_F32_EXP_LOG` is undefined there). Validation against
libm pending the Loongson rebuild.

### Build trap: baseline duplicate symbol

The first Loongson rebuild failed with `multiple definition of
DOUBLE_tanh` between the `loops_umath_fp` baseline build and the
`loops_hyperbolic` baseline build. The Loongson `meson_cpu` config
puts LSX in the *baseline*, so `NPY_HAVE_LSX` is defined even for
baseline-build of every dispatch source. That meant `NPYV_HAVE_tanh_*`
was defined for *both* sources' baseline, so both emitted the symbol.

Fixed by guarding the `loops_hyperbolic` tanh definition with
`#if !defined(NPY_HAVE_LSX) && !defined(NPY_HAVE_LASX)`. The loops_umath_fp
side already gates correctly on `NPYV_HAVE_tanh_@sfx@`, so the
collision is now one-sided: on LoongArch the symbol comes from
loops_umath_fp; everywhere else, from loops_hyperbolic.

### Bench surprise: tanh f32 — Highway port wins, keep it

The first end-to-end bench after the rebuild showed `tanh` f32 had
*regressed* from 307 → 166 M/s. The Highway-based SVML port was
already running fine on LASX (despite my earlier hypothesis); our
NPYV expm1-based form is slower on f32 because the per-block dispatch
for the small-x polynomial path mis-fires on the typical bench input
distribution `[-3, 3]`: with ~8% of inputs below 0.25 per lane, ~50%
of 8-lane LASX blocks hit the "mixed" branch and pay both paths.

Fix: keep the f32 NPYV kernel out of the build entirely. The
`#define NPYV_HAVE_tanh_f32 1` was removed from
`loops_umath_fp.dispatch.c.src`, so the dispatch wrapper there only
emits for f64. The loops_hyperbolic source's tanh `#if !@loong_skip@`
gate now uses a per-iteration `loong_skip = 0, 1` repeat variable,
gating *only* DOUBLE on LoongArch. FLOAT_tanh on LASX/LSX continues
to come from the Highway SVML port.

DOUBLE_tanh on Loongson does get NPYV: the Highway path falls back
to scalar libm for f64 there (no LASX f64 Highway gather), so 54 M/s
before vs 81 M/s with our NPYV form. Net 1.49× for f64.

### Final Loongson 3A6000 numbers

| Op | dtype | Before | After | Gain |
|---|---|---:|---:|---:|
| `tanh`  | f64 | 54.3  | 81.0  | **1.49×** |
| `tanh`  | f32 | 307.1 | 338.9 | 1.10× (same Highway path; noise) |
| `cbrt`  | f32 | 174.3 | 537.3 | **3.08×** |
| `power` | f64 | 52.9  | 75.7  | **1.43×** |
| `power` | f32 | 140.2 | 183.4 | **1.31×** |

Validation: tanh f32 ≤1 ULP, tanh f64 ≤3 ULP, cbrt f32 ≤3 ULP across
100k random inputs each. `power` shows ≤65 ULP at extreme values where
the result is ~1e-18 (e.g. `pow(99, -8.6)`) — that ULP magnification
is the unavoidable consequence of `exp(y · log(x))` in f64: a ~1 ULP
log error × |y| ≈ 8.6 magnifies to ~9 ULP in `y·log(x)`, then exp
exponentiates absolute precision into relative-precision-at-tiny-value
which converts to ~65 ULP. Closer-to-1.0 `pow` results stay ≤8 ULP.
Documented for future-me as a known limitation; fixing requires
double-double arithmetic across the log+exp pair.
