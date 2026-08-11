/*
 * f32 NPYV kernels for the f16 bridge.
 *
 * loops_umath_fp.dispatch.c.src already contains NPY_FINLINE f32 NPYV
 * kernels for most transcendentals, but they're static (per-TU), so
 * loops_half.dispatch.c.src can't call them directly. Rather than
 * refactor the loops_umath_fp file, this header provides
 * simpler-but-sufficient f32 kernels specifically for f16's precision
 * regime: f16 only needs ~11 bits, so a composition like
 *   sinh(x) = (e^x - e^-x) / 2
 * gives ULP-correct f16 output even though it's slightly looser than
 * the carefully tuned f32 sinh kernel in loops_umath_fp.
 *
 * Pattern:
 *   - Ops whose f32 kernel already ships in npyv_exp_log.h (exp, log,
 *     log1p, expm1) are pulled in via that header.
 *   - The remaining 16 ops are composed here on exp/log, except for
 *     sin/cos/tan/arctan/arcsin/arccos which need direct polynomials.
 *
 * Each function exposed here also gets an NPYV_HAVE_<op>_f32 macro so
 * the loops_half dispatch macro fires for it.
 */
#ifndef _NPY_UMATH_NPYV_F16_KERNELS_H
#define _NPY_UMATH_NPYV_F16_KERNELS_H

#include "npyv_exp_log.h"  /* exp/log/log1p/expm1 + NPYV_HAVE_* macros */

#if NPY_SIMD && defined(__loongarch__) && defined(__loongarch_asx) && defined(NPYV_IMPL_F32_EXP_LOG)

/* ============================================================================
 * Compositions on exp / log
 * ============================================================================ */

#define NPYV_HAVE_sinh_f32 1
NPY_FINLINE npyv_f32 npyv_sinh_FLOAT_kernel(npyv_f32 x)
{
    npyv_f32 ex  = npyv_exp_FLOAT_kernel(x);
    npyv_f32 emx = npyv_exp_FLOAT_kernel(npyv_sub_f32(npyv_zero_f32(), x));
    return npyv_mul_f32(npyv_sub_f32(ex, emx), npyv_setall_f32(0.5f));
}

#define NPYV_HAVE_cosh_f32 1
NPY_FINLINE npyv_f32 npyv_cosh_FLOAT_kernel(npyv_f32 x)
{
    npyv_f32 ex  = npyv_exp_FLOAT_kernel(x);
    npyv_f32 emx = npyv_exp_FLOAT_kernel(npyv_sub_f32(npyv_zero_f32(), x));
    return npyv_mul_f32(npyv_add_f32(ex, emx), npyv_setall_f32(0.5f));
}

/* tanh(x) = 1 - 2/(e^{2x}+1).  For huge |x| the formula returns ±1 with
 * no overflow; for x near 0 the rounding error stays within ULP of f16. */
#define NPYV_HAVE_tanh_f32 1
NPY_FINLINE npyv_f32 npyv_tanh_FLOAT_kernel(npyv_f32 x)
{
    const npyv_f32 one = npyv_setall_f32(1.0f);
    const npyv_f32 two = npyv_setall_f32(2.0f);
    npyv_f32 e2x = npyv_exp_FLOAT_kernel(npyv_mul_f32(two, x));
    return npyv_sub_f32(one, npyv_div_f32(two, npyv_add_f32(e2x, one)));
}

#define NPYV_HAVE_exp2_f32 1
NPY_FINLINE npyv_f32 npyv_exp2_FLOAT_kernel(npyv_f32 x)
{
    return npyv_exp_FLOAT_kernel(npyv_mul_f32(x, npyv_setall_f32(0.6931471805599453f)));
}

#define NPYV_HAVE_log2_f32 1
NPY_FINLINE npyv_f32 npyv_log2_FLOAT_kernel(npyv_f32 x)
{
    return npyv_mul_f32(npyv_log_FLOAT_kernel(x), npyv_setall_f32(1.4426950408889634f));
}

#define NPYV_HAVE_log10_f32 1
NPY_FINLINE npyv_f32 npyv_log10_FLOAT_kernel(npyv_f32 x)
{
    return npyv_mul_f32(npyv_log_FLOAT_kernel(x), npyv_setall_f32(0.4342944819032518f));
}

#define NPYV_HAVE_cbrt_f32 1
NPY_FINLINE npyv_f32 npyv_cbrt_FLOAT_kernel(npyv_f32 x)
{
    const npyv_f32 zero = npyv_zero_f32();
    npyv_f32 abs_x = npyv_abs_f32(x);
    npyv_b32 neg = npyv_cmplt_f32(x, zero);
    npyv_b32 is_zero = npyv_cmpeq_f32(x, zero);
    npyv_f32 abs_safe = npyv_select_f32(is_zero, npyv_setall_f32(1.0f), abs_x);
    npyv_f32 m = npyv_exp_FLOAT_kernel(
        npyv_mul_f32(npyv_log_FLOAT_kernel(abs_safe),
                     npyv_setall_f32(0.3333333333333333f)));
    m = npyv_select_f32(is_zero, zero, m);
    return npyv_select_f32(neg, npyv_sub_f32(zero, m), m);
}

/* arcsinh(x) = log(|x| + sqrt(x²+1)) * sign(x).  No catastrophic
 * cancellation for the f16-input range. */
#define NPYV_HAVE_asinh_f32 1
NPY_FINLINE npyv_f32 npyv_asinh_FLOAT_kernel(npyv_f32 x)
{
    const npyv_f32 one = npyv_setall_f32(1.0f);
    const npyv_f32 zero = npyv_zero_f32();
    npyv_f32 abs_x = npyv_abs_f32(x);
    npyv_f32 t = npyv_sqrt_f32(npyv_muladd_f32(abs_x, abs_x, one));
    npyv_f32 mag = npyv_log_FLOAT_kernel(npyv_add_f32(abs_x, t));
    npyv_b32 neg = npyv_cmplt_f32(x, zero);
    return npyv_select_f32(neg, npyv_sub_f32(zero, mag), mag);
}

/* arccosh(x) = log(x + sqrt(x²-1)).  NaN for x < 1, handled by sqrt. */
#define NPYV_HAVE_acosh_f32 1
NPY_FINLINE npyv_f32 npyv_acosh_FLOAT_kernel(npyv_f32 x)
{
    const npyv_f32 one = npyv_setall_f32(1.0f);
    npyv_f32 t = npyv_sqrt_f32(npyv_sub_f32(npyv_mul_f32(x, x), one));
    return npyv_log_FLOAT_kernel(npyv_add_f32(x, t));
}

/* arctanh(x) = 0.5 * log((1+x)/(1-x)).  For x in (-1, 1), no issues. */
#define NPYV_HAVE_atanh_f32 1
NPY_FINLINE npyv_f32 npyv_atanh_FLOAT_kernel(npyv_f32 x)
{
    const npyv_f32 one = npyv_setall_f32(1.0f);
    npyv_f32 num = npyv_add_f32(one, x);
    npyv_f32 den = npyv_sub_f32(one, x);
    return npyv_mul_f32(npyv_log_FLOAT_kernel(npyv_div_f32(num, den)),
                        npyv_setall_f32(0.5f));
}

/* ============================================================================
 * Direct polynomials (sin / cos / tan / arctan / arcsin / arccos)
 *
 * For f16 inputs (after widening to f32) we have |x| ≤ 65504, so Cody-Waite
 * range reduction works trivially. The polynomial degrees here are sized for
 * f32 precision, which is well beyond what f16's 11-bit mantissa needs.
 * ============================================================================ */

/* Cody-Waite three-part π/2 (signs negated for muladd) — f32-precision */
#define NPYV__F16K_PI2_HI -1.5707963705062866f      /* high 24 bits */
#define NPYV__F16K_PI2_MED 4.371138828673793e-08f   /* next 24 bits */
#define NPYV__F16K_PI2_LO -1.747043124068772e-15f   /* remainder */

/* Returns sin(reduced r) for r in [-π/4, π/4]; FDLIBM f32 coefficients. */
NPY_FINLINE npyv_f32 npyv__sin_poly_f32(npyv_f32 r)
{
    /* sin(r) ≈ r + r³·(S1 + S2·r² + S3·r⁴ + S4·r⁶) */
    const npyv_f32 S1 = npyv_setall_f32(-1.6666654393e-01f);
    const npyv_f32 S2 = npyv_setall_f32( 8.3321608736e-03f);
    const npyv_f32 S3 = npyv_setall_f32(-1.9515295891e-04f);
    const npyv_f32 S4 = npyv_setall_f32( 2.7551809286e-06f);
    npyv_f32 z = npyv_mul_f32(r, r);
    npyv_f32 p = S4;
    p = npyv_muladd_f32(p, z, S3);
    p = npyv_muladd_f32(p, z, S2);
    p = npyv_muladd_f32(p, z, S1);
    return npyv_muladd_f32(npyv_mul_f32(r, z), p, r);
}

/* Returns cos(reduced r) for r in [-π/4, π/4]; FDLIBM f32 coefficients. */
NPY_FINLINE npyv_f32 npyv__cos_poly_f32(npyv_f32 r)
{
    /* cos(r) ≈ 1 - r²·(0.5 - C1·r² - C2·r⁴ - C3·r⁶) */
    const npyv_f32 one = npyv_setall_f32(1.0f);
    const npyv_f32 half = npyv_setall_f32(0.5f);
    const npyv_f32 C1 = npyv_setall_f32(4.1666645854e-02f);
    const npyv_f32 C2 = npyv_setall_f32(-1.3887316255e-03f);
    const npyv_f32 C3 = npyv_setall_f32( 2.4760495502e-05f);
    npyv_f32 z = npyv_mul_f32(r, r);
    npyv_f32 p = C3;
    p = npyv_muladd_f32(p, z, C2);
    p = npyv_muladd_f32(p, z, C1);
    p = npyv_sub_f32(half, npyv_mul_f32(z, p));
    return npyv_sub_f32(one, npyv_mul_f32(z, p));
}

/* Shared reduction + quadrant logic. want_cos=0 returns sin, 1 returns cos. */
NPY_FINLINE npyv_f32 npyv__sincos_f32_via_polys(npyv_f32 x, int want_cos)
{
    const npyv_f32 zero = npyv_zero_f32();
    const npyv_f32 nan  = npyv_setall_f32(NPY_NANF);
    const npyv_f32 inv_pi_2 = npyv_setall_f32(0.6366197723675814f);
    const npyv_f32 pi_2_hi  = npyv_setall_f32(NPYV__F16K_PI2_HI);
    const npyv_f32 pi_2_med = npyv_setall_f32(NPYV__F16K_PI2_MED);
    const npyv_f32 pi_2_lo  = npyv_setall_f32(NPYV__F16K_PI2_LO);

    npyv_b32 not_nan = npyv_notnan_f32(x);
    npyv_f32 x_safe = npyv_select_f32(not_nan, x, zero);

    npyv_f32 y_f = npyv_rint_f32(npyv_mul_f32(x_safe, inv_pi_2));
    npyv_f32 r = npyv_muladd_f32(y_f, pi_2_hi,  x_safe);
    r = npyv_muladd_f32(y_f, pi_2_med, r);
    r = npyv_muladd_f32(y_f, pi_2_lo,  r);

    npyv_f32 sin_r = npyv__sin_poly_f32(r);
    npyv_f32 cos_r = npyv__cos_poly_f32(r);

    npyv_s32 quad = npyv_round_s32_f32(y_f);
    if (want_cos) {
        quad = npyv_add_s32(quad, npyv_setall_s32(1));
    }
    npyv_b32 use_cos = npyv_cmpneq_s32(
        npyv_and_s32(quad, npyv_setall_s32(1)), npyv_setall_s32(0));
    npyv_b32 negate  = npyv_cmpneq_s32(
        npyv_and_s32(quad, npyv_setall_s32(2)), npyv_setall_s32(0));
    npyv_f32 mag = npyv_select_f32(use_cos, cos_r, sin_r);
    npyv_f32 res = npyv_select_f32(negate, npyv_sub_f32(zero, mag), mag);
    return npyv_select_f32(not_nan, res, nan);
}

#define NPYV_HAVE_sin_f32 1
NPY_FINLINE npyv_f32 npyv_sin_FLOAT_kernel(npyv_f32 x)
{ return npyv__sincos_f32_via_polys(x, 0); }

#define NPYV_HAVE_cos_f32 1
NPY_FINLINE npyv_f32 npyv_cos_FLOAT_kernel(npyv_f32 x)
{ return npyv__sincos_f32_via_polys(x, 1); }

/* tan(x) = sin(x)/cos(x) — simple and sufficient for f16 precision */
#define NPYV_HAVE_tan_f32 1
NPY_FINLINE npyv_f32 npyv_tan_FLOAT_kernel(npyv_f32 x)
{
    return npyv_div_f32(npyv__sincos_f32_via_polys(x, 0),
                        npyv__sincos_f32_via_polys(x, 1));
}

/* arctan(x) via a single direct polynomial for |x| ≤ 1, plus the identity
 * arctan(x) = π/2 - arctan(1/x) for |x| > 1. FDLIBM f32 coefficients.
 *
 * SIMD-trap discipline: every lane evaluates every branch, so we have to
 * keep the dividend / divisor / argument of every potentially-trapping
 * op (1/abs_x, sqrt, ordered cmp) safe even for the lanes that will be
 * selected away. NaN is masked to 0 at entry via npyv_notnan_f32 (the
 * one unordered compare LASX exposes) and restored at exit. */
#define NPYV_HAVE_atan_f32 1
NPY_FINLINE npyv_f32 npyv_atan_FLOAT_kernel(npyv_f32 x_in)
{
    const npyv_f32 one  = npyv_setall_f32(1.0f);
    const npyv_f32 zero = npyv_zero_f32();
    const npyv_f32 pi_2 = npyv_setall_f32(1.5707963267948966f);
    const npyv_f32 A0 = npyv_setall_f32( 3.3333334327e-01f);
    const npyv_f32 A1 = npyv_setall_f32(-1.9999158382e-01f);
    const npyv_f32 A2 = npyv_setall_f32( 1.4253635705e-01f);
    const npyv_f32 A3 = npyv_setall_f32(-1.0648017377e-01f);
    const npyv_f32 A4 = npyv_setall_f32( 6.1687607318e-02f);

    /* Mask NaN to 0 so ordered comparisons below don't signal INVALID. */
    npyv_b32 not_nan = npyv_notnan_f32(x_in);
    npyv_f32 x = npyv_select_f32(not_nan, x_in, zero);

    npyv_f32 abs_x = npyv_abs_f32(x);
    npyv_b32 big   = npyv_cmpgt_f32(abs_x, one);
    /* For the "big" branch we'd divide 1/abs_x; for the other lanes
     * abs_x can be 0 and the divide would raise DIVBYZERO even though
     * the result is selected away. Use 1 as the divisor on non-big
     * lanes. */
    npyv_f32 safe_div = npyv_select_f32(big, abs_x, one);
    npyv_f32 inv = npyv_div_f32(one, safe_div);
    npyv_f32 arg = npyv_select_f32(big, inv, abs_x);
    npyv_f32 z = npyv_mul_f32(arg, arg);
    npyv_f32 p = A4;
    p = npyv_muladd_f32(p, z, A3);
    p = npyv_muladd_f32(p, z, A2);
    p = npyv_muladd_f32(p, z, A1);
    p = npyv_muladd_f32(p, z, A0);
    npyv_f32 small_res = npyv_sub_f32(arg, npyv_mul_f32(npyv_mul_f32(arg, z), p));
    npyv_f32 big_res = npyv_sub_f32(pi_2, small_res);
    npyv_f32 mag = npyv_select_f32(big, big_res, small_res);
    npyv_b32 neg = npyv_cmplt_f32(x, zero);
    npyv_f32 res = npyv_select_f32(neg, npyv_sub_f32(zero, mag), mag);
    return npyv_select_f32(not_nan, res, x_in);
}

/* arcsin(x) = arctan(x/sqrt(1-x²)).  Loses precision near |x|=1 but f16's
 * 1 ULP at the boundary is generous. */
#define NPYV_HAVE_asin_f32 1
NPY_FINLINE npyv_f32 npyv_asin_FLOAT_kernel(npyv_f32 x_in)
{
    const npyv_f32 one  = npyv_setall_f32(1.0f);
    const npyv_f32 zero = npyv_zero_f32();
    const npyv_f32 pi_2 = npyv_setall_f32(1.5707963267948966f);
    const npyv_f32 nan_v = npyv_setall_f32(NPY_NANF);

    /* Mask NaN at entry; restore at exit */
    npyv_b32 not_nan = npyv_notnan_f32(x_in);
    npyv_f32 x = npyv_select_f32(not_nan, x_in, zero);
    npyv_f32 abs_x = npyv_abs_f32(x);
    npyv_b32 above_one = npyv_cmpgt_f32(abs_x, one);   /* |x| > 1   → NaN  */
    npyv_b32 at_one    = npyv_cmpeq_f32(abs_x, one);   /* |x| == 1  → ±π/2 */
    /* Force out-of-domain or boundary lanes to 0 so the sqrt argument
     * (1-x²) stays ≥ 1 and the subsequent x/denom stays sane. */
    npyv_f32 x_safe = npyv_select_f32(above_one, zero, x);
    x_safe = npyv_select_f32(at_one, zero, x_safe);
    npyv_f32 denom = npyv_sqrt_f32(npyv_sub_f32(one, npyv_mul_f32(x_safe, x_safe)));
    npyv_f32 arg = npyv_div_f32(x_safe, denom);
    npyv_f32 res = npyv_atan_FLOAT_kernel(arg);
    npyv_b32 neg = npyv_cmplt_f32(x, zero);
    npyv_f32 boundary = npyv_select_f32(neg, npyv_sub_f32(zero, pi_2), pi_2);
    res = npyv_select_f32(at_one,    boundary, res);
    res = npyv_select_f32(above_one, nan_v,    res);
    return npyv_select_f32(not_nan,  res,      x_in);
}

#define NPYV_HAVE_acos_f32 1
NPY_FINLINE npyv_f32 npyv_acos_FLOAT_kernel(npyv_f32 x)
{
    const npyv_f32 pi_2 = npyv_setall_f32(1.5707963267948966f);
    return npyv_sub_f32(pi_2, npyv_asin_FLOAT_kernel(x));
}

/*
 * These bridge kernels do not stay within one f16 ULP across all 65,536
 * input bit patterns. loops_half dispatches them through scalar libm until
 * their approximations meet that requirement.
 */
#define NPYV_F16_FORCE_SCALAR_exp   1
#define NPYV_F16_FORCE_SCALAR_exp2  1
#define NPYV_F16_FORCE_SCALAR_expm1 1
#define NPYV_F16_FORCE_SCALAR_sinh  1
#define NPYV_F16_FORCE_SCALAR_cosh  1
#define NPYV_F16_FORCE_SCALAR_tan   1
#define NPYV_F16_FORCE_SCALAR_asin  1
#define NPYV_F16_FORCE_SCALAR_acos  1
#define NPYV_F16_FORCE_SCALAR_atan  1
#define NPYV_F16_FORCE_SCALAR_tanh  1
#define NPYV_F16_FORCE_SCALAR_acosh 1

#endif /* NPYV_IMPL_F32_EXP_LOG */

#endif /* _NPY_UMATH_NPYV_F16_KERNELS_H */
