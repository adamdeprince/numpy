/*
 * NPYV f64 sin/cos kernels (FDLIBM polynomial + Cody-Waite / Payne-Hanek).
 *
 * Background: numpy's upstream loops_trigonometric.dispatch.cpp disables the
 * f64 SIMD path entirely and routes DOUBLE_sin/DOUBLE_cos to scalar libm
 * (see https://mail.python.org/archives/list/numpy-discussion@python.org/
 *  thread/C6EYZZSR4EWGVKHAZXLE7IBILRMNVK7L/). The concern was that the
 * SVML kernel that shipped before lost precision for large |x| where
 * Cody-Waite range reduction breaks down, and there was no Payne-Hanek
 * fallback.
 *
 * This header provides:
 *   - npyv__sincos_f64_kernel(x, want_cos): full SIMD path using
 *     Cody-Waite reduction; correct only for |x| < 2^20 (the dispatcher
 *     gates on this per-block).
 *   - npyv__sincos_f64_polynomial(r, quad, want_cos): the polynomial +
 *     quadrant-selection half of the kernel, taking r ∈ [-π/4, π/4]
 *     and a quadrant integer as input. Reused for both the Cody-Waite
 *     fast path and the Payne-Hanek slow path.
 *   - npyv__sincos_f64_slow_path(x, want_cos): the slow path. Does
 *     per-lane Payne-Hanek reduction (scalar) to produce r and
 *     quadrant, then runs the polynomial in SIMD. Covers the full f64
 *     range; replaces what used to be a scalar libm fallback.
 *
 * Coefficients are FDLIBM's __kernel_sin / __kernel_cos
 * (sysdeps/ieee754/dbl-64/k_sin.c, k_cos.c). The reduction is FDLIBM-style
 * Cody-Waite with high/medium/low parts of π/2 for the fast path; the
 * slow path uses the FDLIBM __kernel_rem_pio2 algorithm (see
 * payne_hanek_f64.h).
 */
#ifndef _NPY_UMATH_NPYV_SINCOS_H
#define _NPY_UMATH_NPYV_SINCOS_H

#include "npyv_exp_log.h"  /* for NPYV_IMPL_F64_EXP_LOG and s64/f64 cast helpers */
#include "payne_hanek_f64.h"
#include "payne_hanek_simd_table.h"

#if defined(NPYV_IMPL_F64_EXP_LOG)

/* Cody-Waite range threshold. Beyond this the high/med/low π/2 split
 * starts losing accuracy because y = round(x · 2/π) overflows the
 * 53-bit mantissa precision of a single double. The dispatcher uses
 * this to gate between the SIMD fast path and the Payne-Hanek slow
 * path. */
#define NPYV_SINCOS_F64_CODY_MAX (1048576.0)   /* 2^20 */

/* op codes for the unified sincos kernel */
#define NPYV_SINCOS_OP_SIN 0
#define NPYV_SINCOS_OP_COS 1

/*
 * Polynomial + quadrant selection. Takes:
 *   r: reduced argument vector, each lane in [-π/4, π/4]
 *   quad: integer quadrant per lane (0..3 for sin; 0..3 for cos, with
 *         the caller having already added 1 if want_cos is set, OR
 *         passing want_cos and letting this function add it)
 *   want_cos: 0 → return sin(x); 1 → return cos(x)
 *
 * Quadrant convention for sin:
 *   quad%4=0:  sin_r        - bit0=0, bit1=0
 *   quad%4=1:  cos_r        - bit0=1, bit1=0
 *   quad%4=2: -sin_r        - bit0=0, bit1=1
 *   quad%4=3: -cos_r        - bit0=1, bit1=1
 * For cos, the same logic with quad' = quad + 1 (handled here).
 */
NPY_FINLINE npyv_f64
npyv__sincos_f64_polynomial(npyv_f64 r, npyv_s64 quad, int want_cos)
{
    const npyv_f64 zero = npyv_zero_f64();
    const npyv_f64 one  = npyv_setall_f64(1.0);
    const npyv_f64 half = npyv_setall_f64(0.5);

    /* FDLIBM sin polynomial coefficients (k_sin.c) */
    const npyv_f64 S1 = npyv_setall_f64(-1.66666666666666324348e-01);
    const npyv_f64 S2 = npyv_setall_f64( 8.33333333332248946124e-03);
    const npyv_f64 S3 = npyv_setall_f64(-1.98412698298579493134e-04);
    const npyv_f64 S4 = npyv_setall_f64( 2.75573137070700676789e-06);
    const npyv_f64 S5 = npyv_setall_f64(-2.50507602534068634195e-08);
    const npyv_f64 S6 = npyv_setall_f64( 1.58969099521155010221e-10);

    /* FDLIBM cos polynomial coefficients (k_cos.c) */
    const npyv_f64 C1 = npyv_setall_f64( 4.16666666666666019037e-02);
    const npyv_f64 C2 = npyv_setall_f64(-1.38888888888741095749e-03);
    const npyv_f64 C3 = npyv_setall_f64( 2.48015872894767294178e-05);
    const npyv_f64 C4 = npyv_setall_f64(-2.75573143513906633035e-07);
    const npyv_f64 C5 = npyv_setall_f64( 2.08757232129817482790e-09);
    const npyv_f64 C6 = npyv_setall_f64(-1.13596475577881948265e-11);

    npyv_f64 z = npyv_mul_f64(r, r);

    /* sin(r) ≈ r + r·z·(S1 + z·(S2 + z·(S3 + z·(S4 + z·(S5 + z·S6))))) */
    npyv_f64 sp = S6;
    sp = npyv_muladd_f64(sp, z, S5);
    sp = npyv_muladd_f64(sp, z, S4);
    sp = npyv_muladd_f64(sp, z, S3);
    sp = npyv_muladd_f64(sp, z, S2);
    sp = npyv_muladd_f64(sp, z, S1);
    npyv_f64 sin_r = npyv_muladd_f64(npyv_mul_f64(r, z), sp, r);

    /* cos(r) ≈ 1 - z·(0.5 - z·cp) — leading 1 preserved, no catastrophic
     * cancellation for r near 0. */
    npyv_f64 cp = C6;
    cp = npyv_muladd_f64(cp, z, C5);
    cp = npyv_muladd_f64(cp, z, C4);
    cp = npyv_muladd_f64(cp, z, C3);
    cp = npyv_muladd_f64(cp, z, C2);
    cp = npyv_muladd_f64(cp, z, C1);
    npyv_f64 cos_r = npyv_sub_f64(one,
        npyv_mul_f64(z, npyv_sub_f64(half, npyv_mul_f64(z, cp))));

    /* For cos, shift quadrant by 1 (cos(x) = sin(x + π/2)). */
    if (want_cos) {
        quad = npyv_add_s64(quad, npyv_setall_s64(1));
    }
    const npyv_s64 izero  = npyv_setall_s64(0);
    const npyv_s64 ione   = npyv_setall_s64(1);
    const npyv_s64 itwo   = npyv_setall_s64(2);
    npyv_b64 use_cos_mask = npyv_cmpneq_s64(npyv_and_s64(quad, ione), izero);
    npyv_b64 negate_mask  = npyv_cmpneq_s64(npyv_and_s64(quad, itwo), izero);

    npyv_f64 mag = npyv_select_f64(use_cos_mask, cos_r, sin_r);
    return npyv_select_f64(negate_mask, npyv_sub_f64(zero, mag), mag);
}

/*
 * SIMD fast path: Cody-Waite reduction + polynomial. Correct only when
 * every lane satisfies |x| < 2^20; caller is responsible for the gate.
 */
NPY_FINLINE npyv_f64
npyv__sincos_f64_kernel(npyv_f64 x_in, int want_cos)
{
    const npyv_f64 zero = npyv_zero_f64();
    const npyv_f64 nan  = npyv_setall_f64(NPY_NAN);

    /* 2/π and three-part π/2 (signs negated so we can use MulAdd) */
    const npyv_f64 inv_pi_2 = npyv_setall_f64(0x1.45f306dc9c883p-1);
    const npyv_f64 pi_2_hi  = npyv_setall_f64(-0x1.921fb54442d18p+0);
    const npyv_f64 pi_2_med = npyv_setall_f64(-0x1.1a62633145c00p-54);
    const npyv_f64 pi_2_lo  = npyv_setall_f64(-0x1.c1cd129024e09p-110);

    /* Mask NaN to 0 so neither rint nor the polynomial signals FE_INVALID */
    npyv_b64 not_nan = npyv_notnan_f64(x_in);
    npyv_f64 x = npyv_select_f64(not_nan, x_in, zero);

    npyv_f64 y_f = npyv_rint_f64(npyv_mul_f64(x, inv_pi_2));
    npyv_f64 r = npyv_muladd_f64(y_f, pi_2_hi,  x);
    r = npyv_muladd_f64(y_f, pi_2_med, r);
    r = npyv_muladd_f64(y_f, pi_2_lo,  r);

    npyv_s64 quad = npyv__cvt_s64_f64(y_f);
    npyv_f64 result = npyv__sincos_f64_polynomial(r, quad, want_cos);
    return npyv_select_f64(not_nan, result, nan);
}

/*
 * SIMD Payne-Hanek reduction (variant C, SLEEF-style pre-aligned table).
 *
 * Algorithm port of SLEEF's `vrempi_vd2_vd` / `mkrempitab.c` (Naoki
 * Shibata, Boost Software License 1.0). For each lane:
 *
 *   1. Compute ilogb(x) per lane, table index = ilogb(x) - 55.
 *   2. If ilogb(x) > 700, scale x by 2^-64 (table chunks are pre-
 *      scaled by 2^64 to compensate — keeps DD hi from overflowing).
 *   3. Per-lane scalar gather of 4 doubles from the table.
 *   4. Four DD multiplications x * c_i (one per chunk, vectorized).
 *   5. DD-sum the 4 partial products (vectorized).
 *   6. Take fractional part (subtract round(hi)).
 *   7. Multiply by 4 to get quadrant + reduced argument.
 *   8. Multiply remainder by π/2 (DD) for the final reduced argument.
 *
 * The polynomial after the reduction stays vectorized in
 * npyv__sincos_f64_polynomial. Net effect: full SIMD slow path with
 * libm-equivalent precision across the entire f64 normal range.
 */
NPY_FINLINE void
npyv__sincos_f64_simd_ph_reduce(npyv_f64 x, npyv_f64 *r_out, npyv_s64 *quad_out)
{
    const npyv_f64 zero = npyv_zero_f64();
    const npyv_f64 four = npyv_setall_f64(4.0);
    /* Strip sign for ilogb extraction; sign handled separately. */
    const npyv_u64 sign_mask = npyv_setall_u64(0x8000000000000000ULL);
    const npyv_u64 abs_mask  = npyv_setall_u64(0x7FFFFFFFFFFFFFFFULL);
    npyv_u64 x_bits  = npyv_reinterpret_u64_f64(x);
    npyv_u64 sign_b  = npyv_and_u64(x_bits, sign_mask);
    npyv_u64 abs_b   = npyv_and_u64(x_bits, abs_mask);
    npyv_s64 ilogb_v = npyv_sub_s64(
        npyv_reinterpret_s64_u64(npyv_shri_u64(abs_b, 52)),
        npyv_setall_s64(1023));

    /* For ilogb > I_SCALE_THRESHOLD (700), scale |x| by 2^SCALE_M (2^-64)
     * via a constant multiply. The table entries for those indices were
     * pre-scaled by 2^64 in the Python generator. */
    const npyv_f64 scale_factor = npyv_setall_f64(0x1.0p-64);
    const npyv_f64 unit         = npyv_setall_f64(1.0);
    npyv_b64 large_mask = npyv_cmpgt_s64(ilogb_v,
                                         npyv_setall_s64(NPYV_PH_SIMD_I_SCALE_THRESHOLD));
    npyv_f64 scale = npyv_select_f64(large_mask, scale_factor, unit);
    npyv_f64 abs_x_scaled = npyv_mul_f64(
        npyv_reinterpret_f64_u64(abs_b), scale);

    /* Re-apply sign to scaled abs(x). */
    npyv_f64 x_scaled = npyv_reinterpret_f64_u64(
        npyv_or_u64(npyv_reinterpret_u64_f64(abs_x_scaled), sign_b));

    /* Table index per lane, clamped to [0, I_MAX - I_MIN - 1]. */
    npyv_s64 idx_v = npyv_sub_s64(ilogb_v, npyv_setall_s64(NPYV_PH_SIMD_I_MIN));
    NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) npy_int64 idx_arr[npyv_nlanes_f64];
    npyv_storea_s64(idx_arr, idx_v);

    NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) double c0_arr[npyv_nlanes_f64];
    NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) double c1_arr[npyv_nlanes_f64];
    NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) double c2_arr[npyv_nlanes_f64];
    NPY_DECL_ALIGNED(NPY_SIMD_WIDTH) double c3_arr[npyv_nlanes_f64];
    const int idx_max = (NPYV_PH_SIMD_I_MAX - NPYV_PH_SIMD_I_MIN) - 1;
    for (int lane = 0; lane < npyv_nlanes_f64; lane++) {
        int k = (int)idx_arr[lane];
        if (k < 0) k = 0;
        if (k > idx_max) k = idx_max;
        c0_arr[lane] = npyv__ph_simd_table[k * 4 + 0];
        c1_arr[lane] = npyv__ph_simd_table[k * 4 + 1];
        c2_arr[lane] = npyv__ph_simd_table[k * 4 + 2];
        c3_arr[lane] = npyv__ph_simd_table[k * 4 + 3];
    }
    npyv_f64 c0 = npyv_loada_f64(c0_arr);
    npyv_f64 c1 = npyv_loada_f64(c1_arr);
    npyv_f64 c2 = npyv_loada_f64(c2_arr);
    npyv_f64 c3 = npyv_loada_f64(c3_arr);

    /* Four DD multiplications: h_i = x_scaled * c_i (rounded),
     * l_i = fma(x_scaled, c_i, -h_i) (exact error). */
    npyv_f64 h0 = npyv_mul_f64(x_scaled, c0);
    npyv_f64 l0 = npyv_muladd_f64(x_scaled, c0, npyv_sub_f64(zero, h0));
    npyv_f64 h1 = npyv_mul_f64(x_scaled, c1);
    npyv_f64 l1 = npyv_muladd_f64(x_scaled, c1, npyv_sub_f64(zero, h1));
    npyv_f64 h2 = npyv_mul_f64(x_scaled, c2);
    npyv_f64 l2 = npyv_muladd_f64(x_scaled, c2, npyv_sub_f64(zero, h2));
    npyv_f64 h3 = npyv_mul_f64(x_scaled, c3);
    npyv_f64 l3 = npyv_muladd_f64(x_scaled, c3, npyv_sub_f64(zero, h3));

    /* DD sum of the 4 partial products. ddadd2 formula:
     *   s = a_hi + b_hi
     *   v = s - a_hi
     *   e = (a_hi - (s - v)) + (b_hi - v) + (a_lo + b_lo)
     */
    npyv_f64 sh = h0, sl = l0;
    /* + (h1, l1) */
    {
        npyv_f64 s = npyv_add_f64(sh, h1);
        npyv_f64 v = npyv_sub_f64(s, sh);
        npyv_f64 e = npyv_add_f64(
            npyv_add_f64(npyv_sub_f64(sh, npyv_sub_f64(s, v)),
                         npyv_sub_f64(h1, v)),
            npyv_add_f64(sl, l1));
        sh = s; sl = e;
    }
    /* + (h2, l2) */
    {
        npyv_f64 s = npyv_add_f64(sh, h2);
        npyv_f64 v = npyv_sub_f64(s, sh);
        npyv_f64 e = npyv_add_f64(
            npyv_add_f64(npyv_sub_f64(sh, npyv_sub_f64(s, v)),
                         npyv_sub_f64(h2, v)),
            npyv_add_f64(sl, l2));
        sh = s; sl = e;
    }
    /* + (h3, l3) */
    {
        npyv_f64 s = npyv_add_f64(sh, h3);
        npyv_f64 v = npyv_sub_f64(s, sh);
        npyv_f64 e = npyv_add_f64(
            npyv_add_f64(npyv_sub_f64(sh, npyv_sub_f64(s, v)),
                         npyv_sub_f64(h3, v)),
            npyv_add_f64(sl, l3));
        sh = s; sl = e;
    }

    /* DD sum is now (sh, sl) ≈ x/(2π) at ~106-bit precision. Take mod 1:
     * subtract round(sh) — gives alpha_hi in [-0.5, 0.5]. */
    npyv_f64 round_sh = npyv_rint_f64(sh);
    npyv_f64 alpha_hi = npyv_sub_f64(sh, round_sh);
    npyv_f64 alpha_lo = sl;

    /* Multiply alpha by 4 (DD) to expand to [-2, 2]. */
    npyv_f64 a4_hi = npyv_mul_f64(alpha_hi, four);
    npyv_f64 a4_lo = npyv_mul_f64(alpha_lo, four);

    /* The DD value a4 = a4_hi + a4_lo is in [-2, 2]; rint the SUM (not
     * a4_hi alone — when alpha_hi == 0 the integer part lives in
     * a4_lo). Both components are small enough that the plain add is
     * precise. */
    npyv_f64 a4_sum = npyv_add_f64(a4_hi, a4_lo);
    npyv_f64 n_f = npyv_rint_f64(a4_sum);
    npyv_s64 n_int = npyv__cvt_s64_f64(n_f);
    npyv_s64 quad = npyv_and_s64(n_int, npyv_setall_s64(3));

    /* Reduced argument: rho = (a4 - n) in [-0.5, 0.5]; r = rho * π/2.
     * Using a DD π/2 for ULP precision. */
    npyv_f64 rho = npyv_add_f64(npyv_sub_f64(a4_hi, n_f), a4_lo);
    const npyv_f64 pi_2_hi = npyv_setall_f64(1.5707963267948966);
    const npyv_f64 pi_2_lo = npyv_setall_f64(6.123233995736766e-17);
    npyv_f64 r = npyv_add_f64(npyv_mul_f64(rho, pi_2_hi),
                              npyv_mul_f64(rho, pi_2_lo));

    *r_out = r;
    *quad_out = quad;
}

/*
 * Slow path: SIMD Payne-Hanek for finite inputs; scalar fallback for
 * inf/NaN/very-small inputs (the SIMD reduction assumes ilogb is well-
 * defined and >= I_MIN; below that the fast Cody-Waite path already
 * handles things).
 */
NPY_FINLINE npyv_f64
npyv__sincos_f64_slow_path(npyv_f64 x_in, int want_cos)
{
    const npyv_f64 inf = npyv_setall_f64(NPY_INFINITY);
    const npyv_f64 ninf = npyv_setall_f64(-NPY_INFINITY);

    /* SIMD PH only valid for finite |x| with ilogb >= I_MIN (= 55).
     * For inf/NaN, return NaN.  For |x| < 2^55 (rare here since the
     * fast-path gate is 2^20 and most fast-path-rejected blocks are
     * still well below 2^55), still goes through SIMD PH — chunks
     * for small i are valid down to ex = -52 in the table. */
    npyv_b64 not_nan = npyv_notnan_f64(x_in);
    npyv_b64 not_inf = npyv_and_b64(npyv_cmplt_f64(x_in, inf),
                                    npyv_cmpgt_f64(x_in, ninf));
    npyv_b64 finite = npyv_and_b64(not_nan, not_inf);

    /* Mask non-finite lanes to a safe value (1.0) before the reduction
     * so the bit-extraction doesn't trip on inf's exponent field. */
    npyv_f64 x_safe = npyv_select_f64(finite, x_in, npyv_setall_f64(1.0));
    npyv_f64 r;
    npyv_s64 quad;
    npyv__sincos_f64_simd_ph_reduce(x_safe, &r, &quad);
    npyv_f64 result = npyv__sincos_f64_polynomial(r, quad, want_cos);

    /* Restore NaN for inf/NaN input lanes. */
    return npyv_select_f64(finite, result, npyv_setall_f64(NPY_NAN));
}

#endif  /* NPYV_IMPL_F64_EXP_LOG */
#endif  /* _NPY_UMATH_NPYV_SINCOS_H */
