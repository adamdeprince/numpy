/*
 * f16 ↔ f32 bridge for LoongArch LASX.
 *
 * LASX provides native f16-to-f32 widening conversions and f32-to-f16
 * narrowing pack. Wrapping our existing f32 NPYV kernels with those
 * conversions turns f16 ufuncs into "load 16 f16 → convert to two f32
 * vectors → run the f32 kernel twice → pack back to 16 f16 → store".
 * The pack/unpack pair costs ~3 cycles per block; everything else is the
 * f32 kernel we already wrote, so f16 ops effectively run at f32 speed.
 *
 * Intrinsic notes:
 *   __lasx_xvfcvtl_s_h(v_h)  → 8 f32 from the LOW 8 f16 elements of v_h
 *   __lasx_xvfcvth_s_h(v_h)  → 8 f32 from the HIGH 8 f16 elements
 *   __lasx_xvfcvt_h_s(hi, lo)→ pack 8 hi f32 + 8 lo f32 into 16 f16
 *
 * `npyv_f16` here is a 256-bit integer vector holding 16 IEEE 754 f16
 * values (interpreted as `npy_half`, a uint16_t alias).
 */
#ifndef _NPY_UMATH_NPYV_F16_BRIDGE_H
#define _NPY_UMATH_NPYV_F16_BRIDGE_H

#include "simd/simd.h"

#if NPY_SIMD && defined(__loongarch__) && defined(__loongarch_asx)

#include <lasxintrin.h>

typedef __m256i npyv_f16;

#define NPYV_NLANES_F16 16

NPY_FINLINE npyv_f16 npyv_load_f16(const npy_half *ptr)
{
    return (__m256i)__lasx_xvld(ptr, 0);
}

NPY_FINLINE void npyv_store_f16(npy_half *ptr, npyv_f16 v)
{
    __lasx_xvst((__m256i)v, ptr, 0);
}

/* Widen the low 8 f16 elements to a full 8-lane f32 vector */
NPY_FINLINE npyv_f32 npyv_f16_to_f32_lo(npyv_f16 v)
{
    return (npyv_f32)__lasx_xvfcvtl_s_h((__m256i)v);
}

/* Widen the high 8 f16 elements to a full 8-lane f32 vector */
NPY_FINLINE npyv_f32 npyv_f16_to_f32_hi(npyv_f16 v)
{
    return (npyv_f32)__lasx_xvfcvth_s_h((__m256i)v);
}

/* Pack two 8-lane f32 vectors into one 16-lane f16 vector.
 * Per the LoongArch convention, the first argument's lanes become the
 * HIGH-indexed half of the result; the second argument's lanes become
 * the LOW-indexed half. So this is `pack(hi_part, lo_part)`. */
NPY_FINLINE npyv_f16 npyv_f32_to_f16(npyv_f32 hi, npyv_f32 lo)
{
    return (__m256i)__lasx_xvfcvt_h_s((__m256)hi, (__m256)lo);
}

/* The repeated pattern: apply a unary f32 NPYV kernel to all 16 f16
 * lanes. The kernel `fn` takes one npyv_f32 and returns one npyv_f32. */
#define NPYV_F16_APPLY_UNARY(v_h, fn) \
    npyv_f32_to_f16(fn(npyv_f16_to_f32_hi(v_h)), fn(npyv_f16_to_f32_lo(v_h)))

/* Same, but with an extra opaque argument the kernel takes through
 * unchanged (e.g. a `want_cos` flag). */
#define NPYV_F16_APPLY_UNARY_ARG(v_h, fn, arg)              \
    npyv_f32_to_f16(fn(npyv_f16_to_f32_hi(v_h), (arg)),     \
                    fn(npyv_f16_to_f32_lo(v_h), (arg)))

#endif /* loongarch + LASX */

#endif /* _NPY_UMATH_NPYV_F16_BRIDGE_H */
