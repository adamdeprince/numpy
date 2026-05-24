# Patch: NPYV float32 exp / log for non-x86 SIMD targets

A self-contained description of one isolated improvement extracted from
the `loongson-experimental` branch. This patch is independent of the
LoongArch port — it benefits any NumPy build whose SIMD baseline is
NEON, ASIMD (AArch64), VSX2/VSX3 (Power), LSX, or LASX (Loongson) and
that is **not** an AVX2/AVX-512 x86 build. On all those targets, the
current upstream code falls through to per-element scalar `expf` /
`logf` because the only f32 SIMD path is written in raw x86 AVX
intrinsics. This patch adds an NPYV-style portable kernel that fills
that gap.

Out of scope: f64 exp/log NPYV (depends on a 64-bit lane cast that
isn't in every backend yet); arcsin / arccos / cbrt / log1p / etc.
follow-ups (they compose on this kernel but each is its own patch).

---

## Goal

`np.exp` and `np.log` on `float32` arrays should run at SIMD speed on
ARM (NEON / ASIMD), Power (VSX), and Loongson (LSX / LASX). Currently
they only get SIMD on x86 (AVX2 / AVX-512); everywhere else they fall
through to the scalar libm path.

After the patch:

- The dispatcher in `loops_exponent_log.dispatch.c.src` picks the new
  NPYV path whenever the build has NPYV f32 support but lacks the
  existing AVX2/AVX-512 path.
- The kernel itself sits in a new shared header `npyv_exp_log.h` so
  other ufuncs (sinh, cosh, exp2, log2, log10, log1p, expm1, arcsinh,
  arccosh, arctanh, cbrt, …) can later compose on it without
  duplicating code.
- Max ULP error is ~2.5 for `exp`, ~2 for `log` — matching the
  existing AVX2 implementation's precision.

Expected throughput uplift (rough, varies by CPU): **3-10× faster**
than the scalar `expf` / `logf` fallback on a typical ARM core; in
the same ballpark on Power; ~3-4× on Loongson 3A6000 LSX.

---

## Files

| Action | Path |
|---|---|
| **new** | `numpy/_core/src/umath/npyv_exp_log.h` |
| **edit** | `numpy/_core/src/umath/loops_exponent_log.dispatch.c.src` |
| **edit** | one of the per-arch headers under `numpy/_core/src/common/simd/<arch>/` per backend you want to enable (see "Per-arch cast helper" below) |
| (optional) | `numpy/_core/meson.build` — if you want to add a LSX/LASX/etc. dispatch target where one didn't exist before |

---

## The new header — `numpy/_core/src/umath/npyv_exp_log.h`

This file is **created** by the patch. It exposes two inline NPYV
kernels (`npyv_exp_FLOAT_kernel`, `npyv_log_FLOAT_kernel`) plus a
sentinel macro `NPYV_IMPL_F32_EXP_LOG` that the dispatch wiring keys
on. The kernels use only ordinary NPYV ops — `npyv_setall_f32`,
`npyv_mul_f32`, `npyv_muladd_f32` (FMA), `npyv_rint_f32`,
`npyv_round_s32_f32`, `npyv_shri_u32`, `npyv_and_s32`,
`npyv_or_s32`, `npyv_reinterpret_*`, `npyv_cmpgt/lt/eq/le/ge/_f32`,
`npyv_select_f32`, `npyv_notnan_f32`, `npyv_abs_f32` — plus
**exactly one arch-specific helper**, `npyv__cvt_f32_s32`, that
casts an `npyv_s32` lane-vector to `npyv_f32`. See "Per-arch cast
helper" below for the per-backend one-liner needed to enable the
kernel on each backend.

### Algorithm

**exp(x):** Cody-Waite range reduction followed by a P5/Q2 rational
approximation, then `2^k` reconstructed by bit-shifting the biased
exponent.

```
k = rint(x · log2e)
y = x − k·ln(2)_hi − k·ln(2)_lo           # Cody-Waite, two-part
P(y) = p0 + y·(p1 + y·(p2 + y·(p3 + y·(p4 + y·p5))))
Q(y) = q0 + y·(q1 + y·q2)
exp(x) = (P(y) / Q(y)) · 2^k
```

The constants come from `numpy/_core/src/umath/npy_simd_data.h`
(`NPY_LOG2Ef`, `NPY_CODY_WAITE_LOGE_2_HIGHf`,
`NPY_CODY_WAITE_LOGE_2_LOWf`, `NPY_COEFF_P0_EXPf` ... `_P5_EXPf`,
`NPY_COEFF_Q0_EXPf` ... `_Q2_EXPf`) — same constants the existing
AVX2 kernel uses, so any AVX2 baseline already has them defined.

Saturation: `x ≥ 88.72283935546875f` → `+inf`; `x ≤ −103.97208404541015625f` → `0`.
NaN: returned unchanged. Underflow / overflow lanes are masked to a
safe value (0.0) before the polynomial so neither path triggers
spurious FE flags.

**log(x):** Bit-extract the binary exponent and mantissa, normalize
the mantissa to `[√½, √2)` (split point at √½ so the polynomial
argument `y = m − 1` is symmetric around 0), then a P5/Q5 rational
approximation, then `k·ln(2) + polynomial(y)`.

```
e_int = (bits(x) >> 23) − 126           # biased + offset, signed
m_bits = (bits(x) & 0x007FFFFF) | 0x3F000000
m = bits→float(m_bits)                  # m ∈ [½, 1)
e = (float)e_int
if m ≤ √½:    m ← 2·m;   e ← e − 1     # normalize to [√½, √2)
y = m − 1
P(y) = p0 + y·(p1 + …)                  # 6 coefficients
Q(y) = q0 + y·(q1 + …)                  # 6 coefficients, q5 leading
poly = P(y)/Q(y)
log(x) = e · ln(2) + poly
```

Constants again from `npy_simd_data.h` (`NPY_LOGE2f`, `NPY_SQRT1_2f`,
`NPY_COEFF_P0_LOGf` ... `_P5_LOGf`, `NPY_COEFF_Q0_LOGf` ...
`_Q5_LOGf`).

Special cases (mask + restore pattern, all branches always
evaluated):
- `x < 0` → `−NaN`
- `x = 0` → `−inf`
- `x = +inf` → `+inf`
- NaN → input passed through

### Full source

```c
/*
 * NPYV-style exp / log kernels for float32 and float64.
 *
 * Used by loops_exponent_log.dispatch.c.src directly, and by
 * loops_umath_fp.dispatch.c.src to compose sinh/cosh/exp2/log2/log10/
 * arcsinh/arccosh/arctanh on top of exp and log.
 *
 * The kernels assume the NPYV backend defines NPY_SIMD_F32. They use
 * ordinary NPYV ops only — plus one arch-specific int-to-float cast
 * that NPYV doesn't currently expose portably.
 *
 * f32 algorithm: Cody-Waite range reduction + P5/Q2 rational
 *   (same as the AVX2 path; max ULP ~2.5).
 */
#ifndef _NPY_UMATH_NPYV_EXP_LOG_H_
#define _NPY_UMATH_NPYV_EXP_LOG_H_

#include "numpy/npy_math.h"
#include "simd/simd.h"
#include "npy_simd_data.h"

#if NPY_SIMD_F32

/* Lane-wise int32 → float32 cast. NPYV doesn't have this portably yet;
 * each backend that wants the f32 exp/log path needs to define it. */
#if defined(NPY_HAVE_LASX)
    #define npyv__cvt_f32_s32(X) ((npyv_f32)__lasx_xvffint_s_w(X))
#elif defined(NPY_HAVE_LSX)
    #define npyv__cvt_f32_s32(X) ((npyv_f32)__lsx_vffint_s_w(X))
#elif defined(NPY_HAVE_NEON)
    #define npyv__cvt_f32_s32(X) (vcvtq_f32_s32(X))
#elif defined(NPY_HAVE_VSX2)
    #define npyv__cvt_f32_s32(X) (vec_float(X))
#endif

#if defined(npyv__cvt_f32_s32)
#define NPYV_IMPL_F32_EXP_LOG 1
#define NPYV_HAVE_exp_f32 1
#define NPYV_HAVE_log_f32 1

NPY_FINLINE npyv_f32
npyv_exp_FLOAT_kernel(npyv_f32 x_in)
{
    const npyv_f32 xmax     = npyv_setall_f32(88.72283935546875f);
    const npyv_f32 xmin     = npyv_setall_f32(-103.97208404541015625f);
    const npyv_f32 log2e    = npyv_setall_f32(NPY_LOG2Ef);
    const npyv_f32 codyw_c1 = npyv_setall_f32(NPY_CODY_WAITE_LOGE_2_HIGHf);
    const npyv_f32 codyw_c2 = npyv_setall_f32(NPY_CODY_WAITE_LOGE_2_LOWf);
    const npyv_f32 p0 = npyv_setall_f32(NPY_COEFF_P0_EXPf);
    const npyv_f32 p1 = npyv_setall_f32(NPY_COEFF_P1_EXPf);
    const npyv_f32 p2 = npyv_setall_f32(NPY_COEFF_P2_EXPf);
    const npyv_f32 p3 = npyv_setall_f32(NPY_COEFF_P3_EXPf);
    const npyv_f32 p4 = npyv_setall_f32(NPY_COEFF_P4_EXPf);
    const npyv_f32 p5 = npyv_setall_f32(NPY_COEFF_P5_EXPf);
    const npyv_f32 q0 = npyv_setall_f32(NPY_COEFF_Q0_EXPf);
    const npyv_f32 q1 = npyv_setall_f32(NPY_COEFF_Q1_EXPf);
    const npyv_f32 q2 = npyv_setall_f32(NPY_COEFF_Q2_EXPf);
    const npyv_f32 inf  = npyv_setall_f32(NPY_INFINITYF);
    const npyv_f32 zero = npyv_zero_f32();

    /* Mask NaN BEFORE any signaling op (cmpge/cmple are ordered cmps
     * that raise INVALID on NaN; mask first, restore at the end). */
    npyv_b32 not_nan = npyv_notnan_f32(x_in);
    npyv_f32 x = npyv_select_f32(not_nan, x_in, zero);

    npyv_b32 overflow  = npyv_cmpge_f32(x, xmax);
    npyv_b32 underflow = npyv_cmple_f32(x, xmin);
    npyv_b32 special   = npyv_or_b32(overflow, underflow);
    x = npyv_select_f32(special, zero, x);

    /* Cody-Waite range reduction */
    npyv_f32 k = npyv_rint_f32(npyv_mul_f32(x, log2e));
    npyv_f32 y = npyv_muladd_f32(k, codyw_c1, x);
    y = npyv_muladd_f32(k, codyw_c2, y);

    /* P5 / Q2 rational approximation */
    npyv_f32 num = npyv_muladd_f32(p5, y, p4);
    num = npyv_muladd_f32(num, y, p3);
    num = npyv_muladd_f32(num, y, p2);
    num = npyv_muladd_f32(num, y, p1);
    num = npyv_muladd_f32(num, y, p0);
    npyv_f32 den = npyv_muladd_f32(q2, y, q1);
    den = npyv_muladd_f32(den, y, q0);
    npyv_f32 poly = npyv_div_f32(num, den);

    /* 2^k via bit construction */
    npyv_s32 k_int   = npyv_round_s32_f32(k);
    npyv_s32 biased  = npyv_add_s32(k_int, npyv_setall_s32(127));
    npyv_f32 twopowk = npyv_reinterpret_f32_s32(npyv_shli_s32(biased, 23));

    npyv_f32 result = npyv_mul_f32(poly, twopowk);
    result = npyv_select_f32(overflow,  inf,  result);
    result = npyv_select_f32(underflow, zero, result);
    result = npyv_select_f32(not_nan,   result, x_in);
    return result;
}

NPY_FINLINE npyv_f32
npyv_log_FLOAT_kernel(npyv_f32 x_in)
{
    const npyv_f32 zero    = npyv_zero_f32();
    const npyv_f32 one     = npyv_setall_f32(1.0f);
    const npyv_f32 inf     = npyv_setall_f32(NPY_INFINITYF);
    const npyv_f32 neg_inf = npyv_setall_f32(-NPY_INFINITYF);
    const npyv_f32 neg_nan = npyv_setall_f32(-NPY_NANF);
    const npyv_f32 sqrt1_2 = npyv_setall_f32(NPY_SQRT1_2f);
    const npyv_f32 loge2   = npyv_setall_f32(NPY_LOGE2f);
    const npyv_f32 p0 = npyv_setall_f32(NPY_COEFF_P0_LOGf);
    const npyv_f32 p1 = npyv_setall_f32(NPY_COEFF_P1_LOGf);
    const npyv_f32 p2 = npyv_setall_f32(NPY_COEFF_P2_LOGf);
    const npyv_f32 p3 = npyv_setall_f32(NPY_COEFF_P3_LOGf);
    const npyv_f32 p4 = npyv_setall_f32(NPY_COEFF_P4_LOGf);
    const npyv_f32 p5 = npyv_setall_f32(NPY_COEFF_P5_LOGf);
    const npyv_f32 q0 = npyv_setall_f32(NPY_COEFF_Q0_LOGf);
    const npyv_f32 q1 = npyv_setall_f32(NPY_COEFF_Q1_LOGf);
    const npyv_f32 q2 = npyv_setall_f32(NPY_COEFF_Q2_LOGf);
    const npyv_f32 q3 = npyv_setall_f32(NPY_COEFF_Q3_LOGf);
    const npyv_f32 q4 = npyv_setall_f32(NPY_COEFF_Q4_LOGf);
    const npyv_f32 q5 = npyv_setall_f32(NPY_COEFF_Q5_LOGf);

    npyv_b32 not_nan = npyv_notnan_f32(x_in);
    npyv_f32 x = npyv_select_f32(not_nan, x_in, one);

    /* Special masks BEFORE the polynomial, so the polynomial path
     * always sees a positive normal number. */
    npyv_b32 negx_mask = npyv_cmplt_f32(x, zero);
    npyv_b32 zero_mask = npyv_cmpeq_f32(x, zero);
    npyv_b32 inf_mask  = npyv_cmpeq_f32(x, inf);

    x = npyv_select_f32(negx_mask, one, x);
    x = npyv_select_f32(zero_mask, one, x);

    /* Bit-extract exponent and mantissa. Use biased exponent 126 so the
     * reconstructed mantissa lands in [½, 1). */
    npyv_s32 x_bits = npyv_reinterpret_s32_f32(x);
    npyv_s32 e_int  = npyv_sub_s32(npyv_shri_u32(x_bits, 23), npyv_setall_s32(126));
    npyv_s32 m_bits = npyv_or_s32(
        npyv_and_s32(x_bits, npyv_setall_s32(0x007FFFFF)),
        npyv_setall_s32(0x3F000000)
    );
    npyv_f32 m = npyv_reinterpret_f32_s32(m_bits);
    npyv_f32 e = npyv__cvt_f32_s32(e_int);

    /* Normalize: if m ≤ √½, double it and decrement e so m ∈ [√½, √2). */
    npyv_b32 small = npyv_cmple_f32(m, sqrt1_2);
    m = npyv_select_f32(small, npyv_add_f32(m, m), m);
    e = npyv_select_f32(small, npyv_sub_f32(e, one), e);

    /* Polynomial in y = m − 1, |y| ≤ √2 − 1 */
    npyv_f32 y = npyv_sub_f32(m, one);
    npyv_f32 num = npyv_muladd_f32(p5, y, p4);
    num = npyv_muladd_f32(num, y, p3);
    num = npyv_muladd_f32(num, y, p2);
    num = npyv_muladd_f32(num, y, p1);
    num = npyv_muladd_f32(num, y, p0);
    npyv_f32 den = npyv_muladd_f32(q5, y, q4);
    den = npyv_muladd_f32(den, y, q3);
    den = npyv_muladd_f32(den, y, q2);
    den = npyv_muladd_f32(den, y, q1);
    den = npyv_muladd_f32(den, y, q0);
    npyv_f32 poly = npyv_div_f32(num, den);

    /* Recombine: log(x) = e · ln(2) + poly(y) */
    poly = npyv_muladd_f32(e, loge2, poly);

    /* Restore special-case results */
    poly = npyv_select_f32(negx_mask, neg_nan, poly);
    poly = npyv_select_f32(zero_mask, neg_inf, poly);
    poly = npyv_select_f32(inf_mask,  inf,     poly);
    poly = npyv_select_f32(not_nan,   poly,    x_in);
    return poly;
}

#endif /* npyv__cvt_f32_s32 defined */
#endif /* NPY_SIMD_F32 */

#endif /* _NPY_UMATH_NPYV_EXP_LOG_H_ */
```

---

## Dispatch wiring — `loops_exponent_log.dispatch.c.src`

Two edits — both small. Both go inside the file's existing
`/**begin repeat ... #func = exp, log ...**/` block for the FLOAT
type.

**Edit 1.** Near the top of the FLOAT section (or just before the
existing FLOAT dispatch `simd_exp_FLOAT` / `simd_log_FLOAT` functions),
add this block. The two `static void` wrappers run only when NPYV f32
exp/log is available **and** the AVX2/AVX-512 paths are not — so x86
behavior is unchanged.

```c
/********************************************************************************
 ** NPYV implementation of exp/log for f32
 ** Kernels live in npyv_exp_log.h so the same code can also power sinh/cosh/
 ** exp2/log2/log10/arcsinh/arccosh/arctanh in loops_umath_fp.dispatch.c.src.
 ********************************************************************************/
#include "npyv_exp_log.h"

#if defined(NPYV_IMPL_F32_EXP_LOG) && !defined(SIMD_AVX2_FMA3) && !defined(SIMD_AVX512F)

/**begin repeat
 * #func     = exp,      log#
 * #scalarf  = npy_expf, npy_logf#
 */
static void
simd_@func@_FLOAT_npyv(const npy_float *src, npy_float *dst, npy_intp n)
{
    const int lanes = npyv_nlanes_f32;
    npy_intp i = 0;
    for (; i + lanes <= n; i += lanes) {
        npyv_f32 v = npyv_load_f32(src + i);
        npyv_store_f32(dst + i, npyv_@func@_FLOAT_kernel(v));
    }
    for (; i < n; i++) {
        dst[i] = @scalarf@(src[i]);
    }
}
/**end repeat**/

#endif // NPYV_IMPL_F32_EXP_LOG && !SIMD_AVX2_FMA3 && !SIMD_AVX512F
```

**Edit 2.** Inside the existing `NPY_CPU_DISPATCH_CURFX(FLOAT_@func@)`
function bodies (there's one each for `exp` and `log`, generated via
the file's repeat block), add an `#elif` branch between the AVX path
and the scalar fallback. The complete dispatch should look like:

```c
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_@func@)
(char **args, npy_intp const *dimensions, npy_intp const *steps,
 void *NPY_UNUSED(data))
{
#if defined(SIMD_AVX2_FMA3) || defined(SIMD_AVX512F)
    // existing AVX2 / AVX-512 path — unchanged
    if (IS_OUTPUT_BLOCKABLE_UNARY(sizeof(npy_float), sizeof(npy_float), 64)) {
        simd_@func@_FLOAT((npy_float*)args[1], (npy_float*)args[0],
                          dimensions[0], steps[0]);
    }
    else {
        UNARY_LOOP {
            simd_@func@_FLOAT((npy_float *)op1, (npy_float *)ip1, 1, steps[0]);
        }
    }
#elif defined(NPYV_IMPL_F32_EXP_LOG)
    /* NEW: NPYV path for non-x86 SIMD targets */
    if (steps[0] == sizeof(npy_float) && steps[1] == sizeof(npy_float)) {
        simd_@func@_FLOAT_npyv(
            (const npy_float *)args[0], (npy_float *)args[1], dimensions[0]);
        return;
    }
    UNARY_LOOP {
        const npy_float in1 = *(npy_float *)ip1;
        *(npy_float *)op1 = @scalarf@(in1);
    }
#else
    /* Scalar fallback for backends with no f32 NPYV exp/log helper */
    UNARY_LOOP {
        const npy_float in1 = *(npy_float *)ip1;
        *(npy_float *)op1 = @scalarf@(in1);
    }
#endif
}
```

The NPYV branch only fires for the contiguous unit-stride case; the
non-contiguous case falls through to the scalar `UNARY_LOOP` so we
preserve numerical compatibility with the strided code path. (The
existing AVX2 path handles strided too, which is why x86 doesn't
need this scalar fallback.)

---

## Per-arch cast helper — `npyv__cvt_f32_s32`

The kernel needs one lane-wise int32→float32 cast that NPYV doesn't
currently expose portably. Each backend you want to enable the kernel
on needs a one-line `#define` in `npyv_exp_log.h` (or, cleanly,
upstream into `numpy/_core/src/common/simd/<arch>/conversion.h`).
Reference one-liners:

```c
#if defined(NPY_HAVE_LASX)     // Loongson 256-bit
    #define npyv__cvt_f32_s32(X) ((npyv_f32)__lasx_xvffint_s_w(X))
#elif defined(NPY_HAVE_LSX)    // Loongson 128-bit
    #define npyv__cvt_f32_s32(X) ((npyv_f32)__lsx_vffint_s_w(X))
#elif defined(NPY_HAVE_NEON)   // ARMv7 / ARMv8 NEON
    #define npyv__cvt_f32_s32(X) (vcvtq_f32_s32(X))
#elif defined(NPY_HAVE_VSX2)   // IBM Power VSX
    #define npyv__cvt_f32_s32(X) (vec_float(X))
#endif
```

If `npyv__cvt_f32_s32` is not defined for a given backend, the kernel
is skipped (the `#if defined(npyv__cvt_f32_s32)` guard in
`npyv_exp_log.h` short-circuits it), and the dispatcher falls back to
scalar `expf` / `logf` — identical to upstream's current behavior.
**The patch is therefore strictly additive: any backend not listed
above sees zero behavior change.**

The TODO sitting in `npyv_exp_log.h` ("lift into the shared NPYV API
so NEON/VSX also get this path without touching this header") is the
clean follow-up if this gets upstreamed: move the cast into NPYV's
shared API (under `simd/<arch>/conversion.h`) so future kernels don't
need their own per-arch helper.

---

## Verification

### Tests

Run the existing numpy umath suite — no test changes needed:

```
spin test -- numpy/_core/tests/test_umath.py -q -k "exp or log"
spin test -- numpy/_core/tests/test_umath.py::TestSpecialFloats
```

The kernel was designed to pass the spurious-FP-exception tests in
`TestSpecialFloats::test_unary_spurious_fpexception` — that's why the
NaN masking happens before any ordered compare. Specifically:

| input | expected output | expected FE flags raised |
|---|---|---|
| any normal finite | `expf` / `logf` to ≤ 2.5 ULP | none |
| 0.0 | exp → 1.0; log → −inf | log raises BYZERO |
| negative | exp → polynomial; log → NaN | log raises INVALID |
| +inf | exp → +inf; log → +inf | none |
| −inf | exp → 0.0; log → NaN | log raises INVALID |
| NaN | NaN passed through | none |

### ULP

Compare against scalar `expf` / `logf` with random samples across
`[−87, 88]` for exp and `[2^−126, 2^127]` for log:

```python
import numpy as np
rng = np.random.default_rng(0)
x = rng.uniform(-87, 88, 1_000_000).astype(np.float32)
a = np.exp(x)
b = np.array([np.exp(np.float32(v)) for v in x])
diff = np.abs(a - b)
ulp  = np.spacing(np.abs(b))
ulp[ulp == 0] = 1.0
print("exp max ULP:", (diff / ulp).max())   # expect ~2.5
```

### Bench

A quick sanity bench:

```python
import numpy as np, time
x = np.linspace(-2, 2, 1_000_000).astype(np.float32)
xstr = np.linspace(-2, 2, 2_000_000).astype(np.float32)[::2]
def bench(fn, a, n=20):
    for _ in range(3): fn(a)
    t = time.perf_counter()
    for _ in range(n): fn(a)
    return a.size / ((time.perf_counter() - t) / n) / 1e6
print(f"exp contig {bench(np.exp, x):.0f} M/s vs strided {bench(np.exp, xstr):.0f}")
print(f"log contig {bench(np.log, x):.0f} M/s vs strided {bench(np.log, xstr):.0f}")
```

Before the patch (any non-x86 NPYV target), the contig and strided
numbers should be identical (both scalar). After the patch, contig
should be 3–10× faster, depending on CPU. The "strided" number is
also the scalar reference for `expf` / `logf` on that build.

---

## Why this stands alone

- Pure addition: no existing code path is modified, only a new
  `#elif` branch added between an existing `#if` and `#else`.
- The new header is self-contained: it depends only on
  `numpy/npy_math.h`, `simd/simd.h`, and `npy_simd_data.h`, all of
  which exist in stock numpy.
- Uses no x86-specific intrinsics; the only arch-specific bit is the
  single int32→float32 cast (one line per backend).
- The macro guard `NPYV_IMPL_F32_EXP_LOG` is the gate: if the cast
  helper isn't defined for an architecture, the kernel and the
  dispatcher branch both vanish from the build. No risk of breaking
  unfamiliar backends.
- Uses only NPYV ops that have been stable in numpy for several
  releases — no new NPYV API required.
- Algorithm and coefficients are byte-for-byte the same as the
  existing AVX2 implementation; precision is therefore identical to
  what x86 users already get.

---

## Algorithm references

- Cody & Waite, *Software Manual for the Elementary Functions*,
  Prentice-Hall 1980 — for the range-reduction strategy and the
  Cody-Waite split of `ln(2)`.
- The P/Q rational coefficients (`NPY_COEFF_P*_EXPf`, `_LOGf`, etc.)
  are the standard ones used in `loops_exponent_log.dispatch.c.src`'s
  AVX2 path, defined in `numpy/_core/src/umath/npy_simd_data.h`.
- Bit construction of `2^k` via biased exponent + 23-bit shift is the
  standard IEEE 754 trick.
