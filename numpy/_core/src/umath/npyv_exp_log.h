/*
 * NPYV-style exp / log kernels for float32 and float64.
 *
 * Used by loops_exponent_log.dispatch.c.src directly, and by
 * loops_umath_fp.dispatch.c.src to compose sinh/cosh/exp2/log2/log10/
 * arcsinh/arccosh/arctanh on top of exp and log.
 *
 * The kernels assume the NPYV backend defines NPY_SIMD_F32 and/or
 * NPY_SIMD_F64. They use ordinary NPYV ops only — plus two arch-specific
 * int-to-float casts that NPYV doesn't currently expose portably.
 *
 * f32 algorithm: Cody-Waite range reduction + P5/Q2 rational approximation
 *   (same as the AVX2 path; max ULP ~2.5).
 * f64 algorithm: Cephes rational approximation for exp; FDLIBM-style
 *   polynomial for log (max ULP ~2 for exp, ~1 for log).
 */
#ifndef _NPY_UMATH_NPYV_EXP_LOG_H_
#define _NPY_UMATH_NPYV_EXP_LOG_H_

#include "numpy/npy_math.h"
#include "simd/simd.h"
#include "npy_simd_data.h"

/*
 * Keep these kernels available as building blocks for the f16 bridge and
 * other NPYV kernels, but do not dispatch them directly when they do not
 * meet NumPy's bundled transcendental reference tolerances. The ufunc
 * dispatchers honor these flags and use their scalar libm paths instead.
 */
#define NPYV_FORCE_SCALAR_exp_f32   1
#define NPYV_FORCE_SCALAR_exp_f64   1
#define NPYV_FORCE_SCALAR_log_f32   1
#define NPYV_FORCE_SCALAR_log_f64   1
#define NPYV_FORCE_SCALAR_log1p_f32 1
#define NPYV_FORCE_SCALAR_expm1_f32 1
#define NPYV_FORCE_SCALAR_expm1_f64 1

/* ============================================================================
 * float32 kernel
 * ============================================================================ */
#if NPY_SIMD_F32

// int32 -> float32 lane-wise cast. TODO: lift into the shared NPYV API
// so NEON/VSX also get this path without touching this header.
#if defined(NPY_HAVE_LASX)
    #define npyv__cvt_f32_s32(X) ((npyv_f32)__lasx_xvffint_s_w(X))
#elif defined(NPY_HAVE_LSX)
    #define npyv__cvt_f32_s32(X) ((npyv_f32)__lsx_vffint_s_w(X))
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

    npyv_b32 not_nan = npyv_notnan_f32(x_in);
    npyv_f32 x = npyv_select_f32(not_nan, x_in, zero);

    npyv_b32 overflow  = npyv_cmpge_f32(x, xmax);
    npyv_b32 underflow = npyv_cmple_f32(x, xmin);
    npyv_b32 special = npyv_or_b32(overflow, underflow);
    x = npyv_select_f32(special, zero, x);

    npyv_f32 k = npyv_rint_f32(npyv_mul_f32(x, log2e));
    npyv_f32 y = npyv_muladd_f32(k, codyw_c1, x);
    y = npyv_muladd_f32(k, codyw_c2, y);

    npyv_f32 num = npyv_muladd_f32(p5, y, p4);
    num = npyv_muladd_f32(num, y, p3);
    num = npyv_muladd_f32(num, y, p2);
    num = npyv_muladd_f32(num, y, p1);
    num = npyv_muladd_f32(num, y, p0);
    npyv_f32 den = npyv_muladd_f32(q2, y, q1);
    den = npyv_muladd_f32(den, y, q0);
    npyv_f32 poly = npyv_div_f32(num, den);

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

    npyv_b32 negx_mask = npyv_cmplt_f32(x, zero);
    npyv_b32 zero_mask = npyv_cmpeq_f32(x, zero);
    npyv_b32 inf_mask  = npyv_cmpeq_f32(x, inf);

    x = npyv_select_f32(negx_mask, one, x);
    x = npyv_select_f32(zero_mask, one, x);

    npyv_s32 x_bits = npyv_reinterpret_s32_f32(x);
    npyv_s32 e_int  = npyv_sub_s32(npyv_shri_u32(x_bits, 23), npyv_setall_s32(126));
    npyv_s32 m_bits = npyv_or_s32(
        npyv_and_s32(x_bits, npyv_setall_s32(0x007FFFFF)),
        npyv_setall_s32(0x3F000000)
    );
    npyv_f32 m = npyv_reinterpret_f32_s32(m_bits);
    npyv_f32 e = npyv__cvt_f32_s32(e_int);

    npyv_b32 small = npyv_cmple_f32(m, sqrt1_2);
    m = npyv_select_f32(small, npyv_add_f32(m, m), m);
    e = npyv_select_f32(small, npyv_sub_f32(e, one), e);

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

    poly = npyv_muladd_f32(e, loge2, poly);

    poly = npyv_select_f32(negx_mask, neg_nan, poly);
    poly = npyv_select_f32(zero_mask, neg_inf, poly);
    poly = npyv_select_f32(inf_mask,  inf,     poly);
    poly = npyv_select_f32(not_nan,   poly,    x_in);
    return poly;
}

/*
 * log1p(x) for f32.
 * For |x| < 2^-5 (≈ 0.031), a 5-term Taylor series is accurate to ~f32 eps
 * and avoids the catastrophic (1+x) cancellation that hits when x < eps.
 * For |x| >= 2^-5, fall through to log_kernel(1+x), which has full precision
 * because 1+x rounds to a value with the full information of x.
 *
 * Special cases inherit from log_kernel: x = -1 → -inf, x < -1 → NaN,
 * x = +inf → +inf, NaN → NaN.
 */
#define NPYV_HAVE_log1p_f32 1
NPY_FINLINE npyv_f32 npyv_log1p_FLOAT_kernel(npyv_f32 x)
{
    const npyv_f32 one = npyv_setall_f32(1.0f);
    /* Taylor: x - x²/2 + x³/3 - x⁴/4 + x⁵/5
     *       = x * (1 + x*(a2 + x*(a3 + x*(a4 + x*a5)))) */
    const npyv_f32 a2 = npyv_setall_f32(-1.0f / 2.0f);
    const npyv_f32 a3 = npyv_setall_f32( 1.0f / 3.0f);
    const npyv_f32 a4 = npyv_setall_f32(-1.0f / 4.0f);
    const npyv_f32 a5 = npyv_setall_f32( 1.0f / 5.0f);
    npyv_f32 p = npyv_muladd_f32(a5, x, a4);
    p = npyv_muladd_f32(p, x, a3);
    p = npyv_muladd_f32(p, x, a2);
    p = npyv_muladd_f32(p, x, one);
    npyv_f32 taylor = npyv_mul_f32(x, p);

    npyv_f32 direct = npyv_log_FLOAT_kernel(npyv_add_f32(one, x));

    npyv_f32 abs_x = npyv_abs_f32(x);
    npyv_b32 use_taylor = npyv_cmplt_f32(abs_x, npyv_setall_f32(0x1.0p-5f));
    return npyv_select_f32(use_taylor, taylor, direct);
}

/*
 * expm1(x) for f32: Taylor for small |x| (avoids 1.0 - 1.0 cancellation),
 * exp_kernel(x) - 1 for larger.
 */
#define NPYV_HAVE_expm1_f32 1
NPY_FINLINE npyv_f32 npyv_expm1_FLOAT_kernel(npyv_f32 x)
{
    const npyv_f32 one = npyv_setall_f32(1.0f);
    /* Taylor: x + x²/2 + x³/6 + x⁴/24 + x⁵/120
     *       = x * (1 + x*(a2 + x*(a3 + x*(a4 + x*a5)))) */
    const npyv_f32 a2 = npyv_setall_f32(1.0f /   2.0f);
    const npyv_f32 a3 = npyv_setall_f32(1.0f /   6.0f);
    const npyv_f32 a4 = npyv_setall_f32(1.0f /  24.0f);
    const npyv_f32 a5 = npyv_setall_f32(1.0f / 120.0f);
    npyv_f32 p = npyv_muladd_f32(a5, x, a4);
    p = npyv_muladd_f32(p, x, a3);
    p = npyv_muladd_f32(p, x, a2);
    p = npyv_muladd_f32(p, x, one);
    npyv_f32 taylor = npyv_mul_f32(x, p);

    npyv_f32 direct = npyv_sub_f32(npyv_exp_FLOAT_kernel(x), one);

    npyv_f32 abs_x = npyv_abs_f32(x);
    npyv_b32 use_taylor = npyv_cmplt_f32(abs_x, npyv_setall_f32(0x1.0p-5f));
    return npyv_select_f32(use_taylor, taylor, direct);
}

#endif // npyv__cvt_f32_s32 defined
#endif // NPY_SIMD_F32

/* ============================================================================
 * float64 kernel
 * ============================================================================ */
#if NPY_SIMD_F64

// int64 <-> float64 lane-wise casts. TODO: lift into NPYV.
#if defined(NPY_HAVE_LASX)
    #define npyv__cvt_f64_s64(X)  ((npyv_f64)__lasx_xvffint_d_l(X))
    #define npyv__cvt_s64_f64(X)  ((npyv_s64)__lasx_xvftintrne_l_d(X))
#elif defined(NPY_HAVE_LSX)
    #define npyv__cvt_f64_s64(X)  ((npyv_f64)__lsx_vffint_d_l(X))
    #define npyv__cvt_s64_f64(X)  ((npyv_s64)__lsx_vftintrne_l_d(X))
#endif

#if defined(npyv__cvt_f64_s64)
#define NPYV_IMPL_F64_EXP_LOG 1

NPY_FINLINE npyv_f64
npyv_exp_DOUBLE_kernel(npyv_f64 x_in)
{
    const npyv_f64 xmax  = npyv_setall_f64(709.78271289338399673222);
    const npyv_f64 xmin  = npyv_setall_f64(-708.39641853226408);
    const npyv_f64 log2e = npyv_setall_f64(1.4426950408889634073599);
    const npyv_f64 C1    = npyv_setall_f64(6.93145751953125e-1);
    const npyv_f64 C2    = npyv_setall_f64(1.42860682030941723212e-6);
    const npyv_f64 P0 = npyv_setall_f64(1.26177193074810590878e-4);
    const npyv_f64 P1 = npyv_setall_f64(3.02994407707441961300e-2);
    const npyv_f64 P2 = npyv_setall_f64(9.99999999999999999910e-1);
    const npyv_f64 Q0 = npyv_setall_f64(3.00198505138664455042e-6);
    const npyv_f64 Q1 = npyv_setall_f64(2.52448340349684104192e-3);
    const npyv_f64 Q2 = npyv_setall_f64(2.27265548208155028766e-1);
    const npyv_f64 Q3 = npyv_setall_f64(2.00000000000000000009e0);

    const npyv_f64 inf  = npyv_setall_f64(NPY_INFINITY);
    const npyv_f64 zero = npyv_zero_f64();
    const npyv_f64 one  = npyv_setall_f64(1.0);
    const npyv_f64 two  = npyv_setall_f64(2.0);

    npyv_b64 not_nan = npyv_notnan_f64(x_in);
    npyv_f64 x = npyv_select_f64(not_nan, x_in, zero);

    npyv_b64 overflow  = npyv_cmpge_f64(x, xmax);
    npyv_b64 underflow = npyv_cmple_f64(x, xmin);
    npyv_b64 special   = npyv_or_b64(overflow, underflow);
    x = npyv_select_f64(special, zero, x);

    npyv_f64 px = npyv_rint_f64(npyv_mul_f64(x, log2e));
    npyv_f64 xr = npyv_sub_f64(x, npyv_mul_f64(px, C1));
    xr = npyv_sub_f64(xr, npyv_mul_f64(px, C2));

    npyv_f64 xx = npyv_mul_f64(xr, xr);

    npyv_f64 P_xx = npyv_muladd_f64(P0, xx, P1);
    P_xx = npyv_muladd_f64(P_xx, xx, P2);
    npyv_f64 Q_xx = npyv_muladd_f64(Q0, xx, Q1);
    Q_xx = npyv_muladd_f64(Q_xx, xx, Q2);
    Q_xx = npyv_muladd_f64(Q_xx, xx, Q3);

    npyv_f64 xP = npyv_mul_f64(xr, P_xx);
    npyv_f64 z = npyv_div_f64(xP, npyv_sub_f64(Q_xx, xP));
    npyv_f64 mantissa = npyv_muladd_f64(two, z, one);

    npyv_s64 n_int   = npyv__cvt_s64_f64(px);
    npyv_s64 biased  = npyv_add_s64(n_int, npyv_setall_s64(1023));
    npyv_f64 twopown = npyv_reinterpret_f64_s64(npyv_shli_s64(biased, 52));

    npyv_f64 result = npyv_mul_f64(mantissa, twopown);
    result = npyv_select_f64(overflow,  inf,  result);
    result = npyv_select_f64(underflow, zero, result);
    result = npyv_select_f64(not_nan,   result, x_in);
    return result;
}

NPY_FINLINE npyv_f64
npyv_log_DOUBLE_kernel(npyv_f64 x_in)
{
    const npyv_f64 zero    = npyv_zero_f64();
    const npyv_f64 one     = npyv_setall_f64(1.0);
    const npyv_f64 half    = npyv_setall_f64(0.5);
    const npyv_f64 two     = npyv_setall_f64(2.0);
    const npyv_f64 inf     = npyv_setall_f64(NPY_INFINITY);
    const npyv_f64 neg_inf = npyv_setall_f64(-NPY_INFINITY);
    const npyv_f64 neg_nan = npyv_setall_f64(-NPY_NAN);
    const npyv_f64 sqrt1_2 = npyv_setall_f64(NPY_SQRT1_2);
    const npyv_f64 ln2_hi  = npyv_setall_f64(6.93147180369123816490e-1);
    const npyv_f64 ln2_lo  = npyv_setall_f64(1.90821492927058770002e-10);
    const npyv_f64 Lg1 = npyv_setall_f64(6.666666666666735130e-1);
    const npyv_f64 Lg2 = npyv_setall_f64(3.999999999940941908e-1);
    const npyv_f64 Lg3 = npyv_setall_f64(2.857142874366239149e-1);
    const npyv_f64 Lg4 = npyv_setall_f64(2.222219843214978396e-1);
    const npyv_f64 Lg5 = npyv_setall_f64(1.818357216161805012e-1);
    const npyv_f64 Lg6 = npyv_setall_f64(1.531383769920937332e-1);
    const npyv_f64 Lg7 = npyv_setall_f64(1.479819860511658591e-1);

    npyv_b64 not_nan = npyv_notnan_f64(x_in);
    npyv_f64 x = npyv_select_f64(not_nan, x_in, one);

    npyv_b64 negx_mask = npyv_cmplt_f64(x, zero);
    npyv_b64 zero_mask = npyv_cmpeq_f64(x, zero);
    npyv_b64 inf_mask  = npyv_cmpeq_f64(x, inf);

    x = npyv_select_f64(negx_mask, one, x);
    x = npyv_select_f64(zero_mask, one, x);

    npyv_s64 x_bits = npyv_reinterpret_s64_f64(x);
    npyv_s64 e_int  = npyv_sub_s64(npyv_shri_u64(x_bits, 52), npyv_setall_s64(1022));
    npyv_s64 m_bits = npyv_or_s64(
        npyv_and_s64(x_bits, npyv_setall_s64(0x000FFFFFFFFFFFFFLL)),
        npyv_setall_s64(0x3FE0000000000000LL)
    );
    npyv_f64 m = npyv_reinterpret_f64_s64(m_bits);
    npyv_f64 k = npyv__cvt_f64_s64(e_int);

    npyv_b64 small = npyv_cmple_f64(m, sqrt1_2);
    m = npyv_select_f64(small, npyv_add_f64(m, m), m);
    k = npyv_select_f64(small, npyv_sub_f64(k, one), k);

    npyv_f64 f = npyv_sub_f64(m, one);
    npyv_f64 s = npyv_div_f64(f, npyv_add_f64(two, f));
    npyv_f64 z = npyv_mul_f64(s, s);
    npyv_f64 w = npyv_mul_f64(z, z);

    npyv_f64 t1 = npyv_muladd_f64(Lg6, w, Lg4);
    t1 = npyv_muladd_f64(t1, w, Lg2);
    t1 = npyv_mul_f64(t1, w);
    npyv_f64 t2 = npyv_muladd_f64(Lg7, w, Lg5);
    t2 = npyv_muladd_f64(t2, w, Lg3);
    t2 = npyv_muladd_f64(t2, w, Lg1);
    t2 = npyv_mul_f64(t2, z);

    npyv_f64 R = npyv_add_f64(t1, t2);
    npyv_f64 hfsq = npyv_mul_f64(half, npyv_mul_f64(f, f));

    npyv_f64 sR = npyv_muladd_f64(s, npyv_add_f64(hfsq, R), npyv_mul_f64(k, ln2_lo));
    npyv_f64 inner = npyv_sub_f64(npyv_sub_f64(hfsq, sR), f);
    npyv_f64 poly = npyv_sub_f64(npyv_mul_f64(k, ln2_hi), inner);

    poly = npyv_select_f64(negx_mask, neg_nan, poly);
    poly = npyv_select_f64(zero_mask, neg_inf, poly);
    poly = npyv_select_f64(inf_mask,  inf,     poly);
    poly = npyv_select_f64(not_nan,   poly,    x_in);
    return poly;
}

/* log1p f64: per-block dispatch between a fast FDLIBM Padé (Lp1..Lp7) for
 * |x| < sqrt(2)-1 ≈ 0.414 and a "full" path = log_kernel(1+x) plus a
 * correction term that captures the rounding error of 1+x. The kernel
 * branches at the block level via npyv_all_b64, so blocks of small inputs
 * skip log_kernel entirely. For the bench range |x| < 0.5 most blocks fall
 * into the fast path. */
NPY_FINLINE npyv_f64 npyv__log1p_pade_f64(npyv_f64 x)
{
    /* FDLIBM Lp coefficients = 2/(2k+1) for k=1..7 — same as log_kernel's
     * Lg1..Lg7 (the Mercator series for log((1+s)/(1-s))). */
    const npyv_f64 Lp1 = npyv_setall_f64(6.666666666666735130e-01);
    const npyv_f64 Lp2 = npyv_setall_f64(3.999999999940941908e-01);
    const npyv_f64 Lp3 = npyv_setall_f64(2.857142874366239149e-01);
    const npyv_f64 Lp4 = npyv_setall_f64(2.222219843214978396e-01);
    const npyv_f64 Lp5 = npyv_setall_f64(1.818357216161805012e-01);
    const npyv_f64 Lp6 = npyv_setall_f64(1.531383769920937332e-01);
    const npyv_f64 Lp7 = npyv_setall_f64(1.479819860511658591e-01);
    const npyv_f64 two  = npyv_setall_f64(2.0);
    const npyv_f64 half = npyv_setall_f64(0.5);

    /* s = x/(2+x), z = s², hfsq = x²/2
     * log1p(x) = x - (hfsq - s·(hfsq + z·polynomial(z))) */
    npyv_f64 s = npyv_div_f64(x, npyv_add_f64(two, x));
    npyv_f64 z = npyv_mul_f64(s, s);
    npyv_f64 hfsq = npyv_mul_f64(half, npyv_mul_f64(x, x));
    npyv_f64 R = Lp7;
    R = npyv_muladd_f64(R, z, Lp6);
    R = npyv_muladd_f64(R, z, Lp5);
    R = npyv_muladd_f64(R, z, Lp4);
    R = npyv_muladd_f64(R, z, Lp3);
    R = npyv_muladd_f64(R, z, Lp2);
    R = npyv_muladd_f64(R, z, Lp1);
    R = npyv_mul_f64(R, z);
    npyv_f64 inner = npyv_sub_f64(hfsq, npyv_mul_f64(s, npyv_add_f64(hfsq, R)));
    return npyv_sub_f64(x, inner);
}

/* Full-range log1p via log_kernel(1+x) plus correction for the rounding
 * error in the 1+x computation. For |x| > ~0.4 this is the only correct
 * approach (Padé doesn't converge). For tiny |x| the correction gives
 * log1p(x) → x with ULP precision.
 *
 * inf/NaN lanes are masked to 0 before the correction arithmetic to keep
 * inf-inf subtractions from raising spurious INVALID. log_kernel itself
 * is then called on the ACTUAL u = 1+x, which handles inf/-inf/NaN cleanly
 * (returns inf / -NaN / NaN without spurious flags); the correction
 * collapses to 0 for those lanes so the result equals log_kernel's output.
 */
NPY_FINLINE npyv_f64 npyv__log1p_full_f64(npyv_f64 x)
{
    const npyv_f64 zero = npyv_zero_f64();
    const npyv_f64 one  = npyv_setall_f64(1.0);
    const npyv_f64 two  = npyv_setall_f64(2.0);
    const npyv_f64 inf  = npyv_setall_f64(NPY_INFINITY);
    const npyv_f64 ninf = npyv_setall_f64(-NPY_INFINITY);

    /* Finite mask: not inf, not -inf, not NaN */
    npyv_b64 finite = npyv_and_b64(
        npyv_and_b64(npyv_cmplt_f64(x, inf), npyv_cmpgt_f64(x, ninf)),
        npyv_notnan_f64(x));
    /* Mask inf/NaN lanes to 0 for the correction arithmetic; log_kernel
     * still sees the real (1+x). */
    npyv_f64 x_safe = npyv_select_f64(finite, x, zero);
    npyv_f64 u_safe = npyv_add_f64(one, x_safe);

    npyv_b64 u_big = npyv_cmpge_f64(u_safe, two);
    npyv_f64 c_big   = npyv_sub_f64(one, npyv_sub_f64(u_safe, x_safe));
    npyv_f64 c_small = npyv_sub_f64(x_safe, npyv_sub_f64(u_safe, one));
    npyv_f64 c = npyv_select_f64(u_big, c_big, c_small);
    /* When x = -1 exactly, u_safe = 0 and we'd hit 0/0 = NaN in the
     * correction. log_kernel(0) already returns -∞ (the correct log1p(-1)),
     * so just zero out the correction for that lane. */
    npyv_b64 u_zero = npyv_cmpeq_f64(u_safe, zero);
    npyv_f64 u_div = npyv_select_f64(u_zero, one, u_safe);  /* avoid 0/0 */
    npyv_f64 c_div = npyv_select_f64(u_zero, zero, c);
    npyv_f64 correction = npyv_div_f64(c_div, u_div);

    /* log_kernel on the actual 1+x — its own special-case logic handles
     * inf (returns inf), -inf (returns -NaN), 0 (returns -inf), and NaN
     * (passes through). */
    npyv_f64 u = npyv_add_f64(one, x);
    return npyv_add_f64(npyv_log_DOUBLE_kernel(u), correction);
}

#define NPYV_HAVE_log1p_f64 1
NPY_FINLINE npyv_f64 npyv_log1p_DOUBLE_kernel(npyv_f64 x)
{
    /* FDLIBM's k=0 Padé form is accurate to ULP only when u = 1+x lands in
     * [sqrt(2)/2, sqrt(2)) — i.e., x ∈ (1-sqrt(2)/2, sqrt(2)-1) ≈
     * (-0.293, 0.414). The range is ASYMMETRIC (negative bound tighter
     * because (sqrt(2)/2 - 1) ≈ -0.293, not -0.414). Outside this band
     * the formula loses ~14 bits; fall back to log_kernel + correction. */
    const npyv_f64 lo_thresh = npyv_setall_f64(-0.29);
    const npyv_f64 hi_thresh = npyv_setall_f64( 0.41);
    npyv_b64 in_fast = npyv_and_b64(npyv_cmpgt_f64(x, lo_thresh),
                                    npyv_cmplt_f64(x, hi_thresh));
    if (NPY_LIKELY(npyv_all_b64(in_fast))) {
        return npyv__log1p_pade_f64(x);
    }
    return npyv__log1p_full_f64(x);
}

#define NPYV_HAVE_expm1_f64 1
NPY_FINLINE npyv_f64 npyv_expm1_DOUBLE_kernel(npyv_f64 x)
{
    const npyv_f64 one = npyv_setall_f64(1.0);
    const npyv_f64 a2 = npyv_setall_f64(1.0 /    2.0);
    const npyv_f64 a3 = npyv_setall_f64(1.0 /    6.0);
    const npyv_f64 a4 = npyv_setall_f64(1.0 /   24.0);
    const npyv_f64 a5 = npyv_setall_f64(1.0 /  120.0);
    const npyv_f64 a6 = npyv_setall_f64(1.0 /  720.0);
    const npyv_f64 a7 = npyv_setall_f64(1.0 / 5040.0);
    npyv_f64 p = npyv_muladd_f64(a7, x, a6);
    p = npyv_muladd_f64(p, x, a5);
    p = npyv_muladd_f64(p, x, a4);
    p = npyv_muladd_f64(p, x, a3);
    p = npyv_muladd_f64(p, x, a2);
    p = npyv_muladd_f64(p, x, one);
    npyv_f64 taylor = npyv_mul_f64(x, p);

    npyv_f64 direct = npyv_sub_f64(npyv_exp_DOUBLE_kernel(x), one);

    npyv_f64 abs_x = npyv_abs_f64(x);
    npyv_b64 use_taylor = npyv_cmplt_f64(abs_x, npyv_setall_f64(0x1.0p-7));
    return npyv_select_f64(use_taylor, taylor, direct);
}

#endif // npyv__cvt_f64_s64 defined
#endif // NPY_SIMD_F64

#endif // _NPY_UMATH_NPYV_EXP_LOG_H_
