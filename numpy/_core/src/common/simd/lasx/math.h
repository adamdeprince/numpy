#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LASX_MATH_H
#define _NPY_SIMD_LASX_MATH_H

/***************************
 * Elementary
 ***************************/
// Square root
#define npyv_sqrt_f32 __lasx_xvfsqrt_s
#define npyv_sqrt_f64 __lasx_xvfsqrt_d

// Reciprocal
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{ return __lasx_xvfrecip_s(a); }
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{ return __lasx_xvfrecip_d(a); }

// Absolute (clear sign bit)
NPY_FINLINE npyv_f32 npyv_abs_f32(npyv_f32 a)
{ return (npyv_f32)__lasx_xvbitclri_w((__m256i)a, 0x1F); }
NPY_FINLINE npyv_f64 npyv_abs_f64(npyv_f64 a)
{ return (npyv_f64)__lasx_xvbitclri_d((__m256i)a, 0x3F); }

// Square
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return __lasx_xvfmul_s(a, a); }
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return __lasx_xvfmul_d(a, a); }

// Maximum, no NaN handling
#define npyv_max_f32 __lasx_xvfmax_s
#define npyv_max_f64 __lasx_xvfmax_d
NPY_FINLINE npyv_f32 npyv_maxp_f32(npyv_f32 a, npyv_f32 b)
{ return __lasx_xvfmax_s(a, b); }
NPY_FINLINE npyv_f64 npyv_maxp_f64(npyv_f64 a, npyv_f64 b)
{ return __lasx_xvfmax_d(a, b); }
NPY_FINLINE npyv_f32 npyv_maxn_f32(npyv_f32 a, npyv_f32 b)
{
    __m256i mask = __lasx_xvand_v(npyv_notnan_f32(a), npyv_notnan_f32(b));
    __m256 max = __lasx_xvfmax_s(a, b);
    return npyv_select_f32(mask, max, (__m256){NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN});
}
NPY_FINLINE npyv_f64 npyv_maxn_f64(npyv_f64 a, npyv_f64 b)
{
    __m256i mask = __lasx_xvand_v(npyv_notnan_f64(a), npyv_notnan_f64(b));
    __m256d max = __lasx_xvfmax_d(a, b);
    return npyv_select_f64(mask, max, (__m256d){NAN, NAN, NAN, NAN});
}

// Maximum, integer operations
#define npyv_max_u8  __lasx_xvmax_bu
#define npyv_max_s8  __lasx_xvmax_b
#define npyv_max_u16 __lasx_xvmax_hu
#define npyv_max_s16 __lasx_xvmax_h
#define npyv_max_u32 __lasx_xvmax_wu
#define npyv_max_s32 __lasx_xvmax_w
#define npyv_max_u64 __lasx_xvmax_du
#define npyv_max_s64 __lasx_xvmax_d

// Minimum
#define npyv_min_f32 __lasx_xvfmin_s
#define npyv_min_f64 __lasx_xvfmin_d
NPY_FINLINE npyv_f32 npyv_minp_f32(npyv_f32 a, npyv_f32 b)
{ return __lasx_xvfmin_s(a, b); }
NPY_FINLINE npyv_f64 npyv_minp_f64(npyv_f64 a, npyv_f64 b)
{ return __lasx_xvfmin_d(a, b); }
NPY_FINLINE npyv_f32 npyv_minn_f32(npyv_f32 a, npyv_f32 b)
{
    __m256i mask = __lasx_xvand_v(npyv_notnan_f32(a), npyv_notnan_f32(b));
    __m256 min = __lasx_xvfmin_s(a, b);
    return npyv_select_f32(mask, min, (__m256){NAN, NAN, NAN, NAN, NAN, NAN, NAN, NAN});
}
NPY_FINLINE npyv_f64 npyv_minn_f64(npyv_f64 a, npyv_f64 b)
{
    __m256i mask = __lasx_xvand_v(npyv_notnan_f64(a), npyv_notnan_f64(b));
    __m256d min = __lasx_xvfmin_d(a, b);
    return npyv_select_f64(mask, min, (__m256d){NAN, NAN, NAN, NAN});
}

// Minimum, integer operations
#define npyv_min_u8  __lasx_xvmin_bu
#define npyv_min_s8  __lasx_xvmin_b
#define npyv_min_u16 __lasx_xvmin_hu
#define npyv_min_s16 __lasx_xvmin_h
#define npyv_min_u32 __lasx_xvmin_wu
#define npyv_min_s32 __lasx_xvmin_w
#define npyv_min_u64 __lasx_xvmin_du
#define npyv_min_s64 __lasx_xvmin_d

/*
 * Reduce min/max.
 * LASX reduce strategy: first reduce within each 128-bit half (4 lanes f32 ->
 * 1 lane, or 2 lanes f64 -> 1 lane) using xvf{min,max}_s/d with shuffles, then
 * combine the two 128-bit halves with one more vf{min,max}.
 */
#define NPY_IMPL_LASX_REDUCE_MINMAX_FLT(INTRIN, INF, INF64)                                       \
    NPY_FINLINE float npyv_reduce_##INTRIN##_f32(npyv_f32 a)                                      \
    {                                                                                             \
        /* Within each 128-bit half: reduce 4 lanes -> 1 lane (placed at lane 0 of half) */       \
        /* Step A: pairwise reduce (lanes 0-3 vs 2-5...): use shuf to bring high pair down */     \
        __m256 t = __lasx_xvf##INTRIN##_s(a, (__m256)__lasx_xvshuf_w(                             \
            (__m256i)(v8i32){2, 3, 0, 0, 6, 7, 4, 4}, (__m256i)a, (__m256i)a));                   \
        t = __lasx_xvf##INTRIN##_s(t, (__m256)__lasx_xvshuf_w(                                    \
            (__m256i)(v8i32){1, 0, 0, 0, 5, 4, 4, 4}, (__m256i)t, (__m256i)t));                   \
        /* Now t[0] = reduction of low half, t[4] = reduction of high half */                     \
        float lo = t[0], hi = t[4];                                                               \
        return (lo INTRIN##_OP hi) ? lo : hi;                                                     \
    }                                                                                             \
    NPY_FINLINE float npyv_reduce_##INTRIN##n_f32(npyv_f32 a)                                     \
    {                                                                                             \
        npyv_b32 notnan = npyv_notnan_f32(a);                                                     \
        if (NPY_UNLIKELY(!npyv_all_b32(notnan))) {                                                \
            const union { npy_uint32 i; float f;} pnan = {0x7fc00000UL};                          \
            return pnan.f;                                                                        \
        }                                                                                         \
        return npyv_reduce_##INTRIN##_f32(a);                                                     \
    }                                                                                             \
    NPY_FINLINE float npyv_reduce_##INTRIN##p_f32(npyv_f32 a)                                     \
    {                                                                                             \
        npyv_b32 notnan = npyv_notnan_f32(a);                                                     \
        if (NPY_UNLIKELY(!npyv_any_b32(notnan))) { return a[0]; }                                 \
        a = npyv_select_f32(notnan, a, npyv_reinterpret_f32_u32(npyv_setall_u32(INF)));           \
        return npyv_reduce_##INTRIN##_f32(a);                                                     \
    }                                                                                             \
    NPY_FINLINE double npyv_reduce_##INTRIN##_f64(npyv_f64 a)                                     \
    {                                                                                             \
        /* Within each 128-bit half: reduce 2 lanes -> 1, then combine halves. */                 \
        __m256d t = __lasx_xvf##INTRIN##_d(a, (__m256d)__lasx_xvshuf_d(                           \
            (__m256i)(v4i64){1, 0, 3, 2}, (__m256i)a, (__m256i)a));                               \
        double lo = t[0], hi = t[2];                                                              \
        return (lo INTRIN##_OP hi) ? lo : hi;                                                     \
    }                                                                                             \
    NPY_FINLINE double npyv_reduce_##INTRIN##p_f64(npyv_f64 a)                                    \
    {                                                                                             \
        npyv_b64 notnan = npyv_notnan_f64(a);                                                     \
        if (NPY_UNLIKELY(!npyv_any_b64(notnan))) { return a[0]; }                                 \
        a = npyv_select_f64(notnan, a, npyv_reinterpret_f64_u64(npyv_setall_u64(INF64)));         \
        return npyv_reduce_##INTRIN##_f64(a);                                                     \
    }                                                                                             \
    NPY_FINLINE double npyv_reduce_##INTRIN##n_f64(npyv_f64 a)                                    \
    {                                                                                             \
        npyv_b64 notnan = npyv_notnan_f64(a);                                                     \
        if (NPY_UNLIKELY(!npyv_all_b64(notnan))) {                                                \
            const union { npy_uint64 i; double d;} pnan = {0x7ff8000000000000ull};                \
            return pnan.d;                                                                        \
        }                                                                                         \
        return npyv_reduce_##INTRIN##_f64(a);                                                     \
    }

#define min_OP <=
#define max_OP >=
NPY_IMPL_LASX_REDUCE_MINMAX_FLT(min, 0x7f800000, 0x7ff0000000000000ULL)
NPY_IMPL_LASX_REDUCE_MINMAX_FLT(max, 0xff800000, 0xfff0000000000000ULL)
#undef min_OP
#undef max_OP
#undef NPY_IMPL_LASX_REDUCE_MINMAX_FLT

// Integer reduce min/max: do it within each 128-bit half using the same
// pattern as LSX, then combine the two halves with one more scalar op.
// LASX only provides xvpickve2gr for 32-bit and 64-bit lanes, so we extract
// the half/byte we want from the enclosing 32-bit lane and mask/sign-extend.
#define NPY_IMPL_LASX_REDUCE_MINMAX_INT(STYPE, INTRIN, TFLAG, SCALAR_OP)                       \
    NPY_FINLINE STYPE##64 npyv_reduce_##INTRIN##64(__m256i a)                                  \
    {                                                                                          \
        __m256i v64 = npyv_##INTRIN##64(a,                                                     \
            __lasx_xvshuf_d((__m256i)(v4i64){1, 0, 3, 2}, a, a));                              \
        STYPE##64 lo = (STYPE##64)__lasx_xvpickve2gr_d##TFLAG(v64, 0);                         \
        STYPE##64 hi = (STYPE##64)__lasx_xvpickve2gr_d##TFLAG(v64, 2);                         \
        return SCALAR_OP(lo, hi);                                                              \
    }                                                                                          \
    NPY_FINLINE STYPE##32 npyv_reduce_##INTRIN##32(__m256i a)                                  \
    {                                                                                          \
        __m256i v64 = npyv_##INTRIN##32(a,                                                     \
            __lasx_xvshuf_w((__m256i)(v8i32){2, 3, 0, 0, 6, 7, 4, 4}, a, a));                  \
        __m256i v32 = npyv_##INTRIN##32(v64,                                                   \
            __lasx_xvshuf_w((__m256i)(v8i32){1, 0, 0, 0, 5, 4, 4, 4}, v64, v64));              \
        STYPE##32 lo = (STYPE##32)__lasx_xvpickve2gr_w##TFLAG(v32, 0);                         \
        STYPE##32 hi = (STYPE##32)__lasx_xvpickve2gr_w##TFLAG(v32, 4);                         \
        return SCALAR_OP(lo, hi);                                                              \
    }                                                                                          \
    NPY_FINLINE STYPE##16 npyv_reduce_##INTRIN##16(__m256i a)                                  \
    {                                                                                          \
        __m256i v = npyv_##INTRIN##16(a, __lasx_xvbsrl_v(a, 8));                               \
        v = npyv_##INTRIN##16(v, __lasx_xvbsrl_v(v, 4));                                       \
        v = npyv_##INTRIN##16(v, __lasx_xvbsrl_v(v, 2));                                       \
        STYPE##16 lo = (STYPE##16)((npy_uint16)__lasx_xvpickve2gr_wu(v, 0));                   \
        STYPE##16 hi = (STYPE##16)((npy_uint16)__lasx_xvpickve2gr_wu(v, 4));                   \
        return SCALAR_OP(lo, hi);                                                              \
    }                                                                                          \
    NPY_FINLINE STYPE##8 npyv_reduce_##INTRIN##8(__m256i a)                                    \
    {                                                                                          \
        __m256i v = npyv_##INTRIN##8(a, __lasx_xvbsrl_v(a, 8));                                \
        v = npyv_##INTRIN##8(v, __lasx_xvbsrl_v(v, 4));                                        \
        v = npyv_##INTRIN##8(v, __lasx_xvbsrl_v(v, 2));                                        \
        v = npyv_##INTRIN##8(v, __lasx_xvbsrl_v(v, 1));                                        \
        STYPE##8 lo = (STYPE##8)((npy_uint8)__lasx_xvpickve2gr_wu(v, 0));                      \
        STYPE##8 hi = (STYPE##8)((npy_uint8)__lasx_xvpickve2gr_wu(v, 4));                      \
        return SCALAR_OP(lo, hi);                                                              \
    }

#define NPY_LASX_MIN(a, b) ((a) < (b) ? (a) : (b))
#define NPY_LASX_MAX(a, b) ((a) > (b) ? (a) : (b))
NPY_IMPL_LASX_REDUCE_MINMAX_INT(npy_uint, min_u, u, NPY_LASX_MIN)
NPY_IMPL_LASX_REDUCE_MINMAX_INT(npy_int,  min_s, , NPY_LASX_MIN)
NPY_IMPL_LASX_REDUCE_MINMAX_INT(npy_uint, max_u, u, NPY_LASX_MAX)
NPY_IMPL_LASX_REDUCE_MINMAX_INT(npy_int,  max_s, , NPY_LASX_MAX)
#undef NPY_LASX_MIN
#undef NPY_LASX_MAX
#undef NPY_IMPL_LASX_REDUCE_MINMAX_INT

// round to nearest integer even
#define npyv_rint_f32  (__m256)__lasx_xvfrintrne_s
#define npyv_rint_f64  (__m256d)__lasx_xvfrintrne_d
// ceil
#define npyv_ceil_f32  (__m256)__lasx_xvfrintrp_s
#define npyv_ceil_f64  (__m256d)__lasx_xvfrintrp_d
// trunc
#define npyv_trunc_f32 (__m256)__lasx_xvfrintrz_s
#define npyv_trunc_f64 (__m256d)__lasx_xvfrintrz_d
// floor
#define npyv_floor_f32 (__m256)__lasx_xvfrintrm_s
#define npyv_floor_f64 (__m256d)__lasx_xvfrintrm_d

#endif // _NPY_SIMD_LASX_MATH_H
