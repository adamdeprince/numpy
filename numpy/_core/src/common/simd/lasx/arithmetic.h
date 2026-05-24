#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LASX_ARITHMETIC_H
#define _NPY_SIMD_LASX_ARITHMETIC_H

/***************************
 * Addition
 ***************************/
// non-saturated
#define npyv_add_u8  __lasx_xvadd_b
#define npyv_add_s8  __lasx_xvadd_b
#define npyv_add_u16 __lasx_xvadd_h
#define npyv_add_s16 __lasx_xvadd_h
#define npyv_add_u32 __lasx_xvadd_w
#define npyv_add_s32 __lasx_xvadd_w
#define npyv_add_u64 __lasx_xvadd_d
#define npyv_add_s64 __lasx_xvadd_d
#define npyv_add_f32 __lasx_xvfadd_s
#define npyv_add_f64 __lasx_xvfadd_d

// saturated
#define npyv_adds_u8  __lasx_xvsadd_bu
#define npyv_adds_s8  __lasx_xvsadd_b
#define npyv_adds_u16 __lasx_xvsadd_hu
#define npyv_adds_s16 __lasx_xvsadd_h
#define npyv_adds_u32 __lasx_xvsadd_wu
#define npyv_adds_s32 __lasx_xvsadd_w
#define npyv_adds_u64 __lasx_xvsadd_du
#define npyv_adds_s64 __lasx_xvsadd_d

/***************************
 * Subtraction
 ***************************/
// non-saturated
#define npyv_sub_u8  __lasx_xvsub_b
#define npyv_sub_s8  __lasx_xvsub_b
#define npyv_sub_u16 __lasx_xvsub_h
#define npyv_sub_s16 __lasx_xvsub_h
#define npyv_sub_u32 __lasx_xvsub_w
#define npyv_sub_s32 __lasx_xvsub_w
#define npyv_sub_u64 __lasx_xvsub_d
#define npyv_sub_s64 __lasx_xvsub_d
#define npyv_sub_f32 __lasx_xvfsub_s
#define npyv_sub_f64 __lasx_xvfsub_d

// saturated
#define npyv_subs_u8  __lasx_xvssub_bu
#define npyv_subs_s8  __lasx_xvssub_b
#define npyv_subs_u16 __lasx_xvssub_hu
#define npyv_subs_s16 __lasx_xvssub_h
#define npyv_subs_u32 __lasx_xvssub_wu
#define npyv_subs_s32 __lasx_xvssub_w
#define npyv_subs_u64 __lasx_xvssub_du
#define npyv_subs_s64 __lasx_xvssub_d

/***************************
 * Multiplication
 ***************************/
// non-saturated
#define npyv_mul_u8  __lasx_xvmul_b
#define npyv_mul_s8  __lasx_xvmul_b
#define npyv_mul_u16 __lasx_xvmul_h
#define npyv_mul_s16 __lasx_xvmul_h
#define npyv_mul_u32 __lasx_xvmul_w
#define npyv_mul_s32 __lasx_xvmul_w
#define npyv_mul_f32 __lasx_xvfmul_s
#define npyv_mul_f64 __lasx_xvfmul_d

/***************************
 * Integer Division
 ***************************/
NPY_FINLINE npyv_u8 npyv_divc_u8(npyv_u8 a, const npyv_u8x3 divisor)
{
    __m256i mulhi = __lasx_xvmuh_bu(a, divisor.val[0]);
    __m256i q = __lasx_xvsub_b(a, mulhi);
            q = __lasx_xvsrl_b(q, divisor.val[1]);
            q = __lasx_xvadd_b(mulhi, q);
            q = __lasx_xvsrl_b(q, divisor.val[2]);
    return  q;
}
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor);
NPY_FINLINE npyv_s8 npyv_divc_s8(npyv_s8 a, const npyv_s8x3 divisor)
{
    __m256i mulhi = __lasx_xvmuh_b(a, divisor.val[0]);
    __m256i q = __lasx_xvsra_b(__lasx_xvadd_b(a, mulhi), divisor.val[1]);
            q = __lasx_xvsub_b(q, __lasx_xvsrai_b(a, 7));
            q = __lasx_xvsub_b(__lasx_xvxor_v(q, divisor.val[2]), divisor.val[2]);
    return q;
}
NPY_FINLINE npyv_u16 npyv_divc_u16(npyv_u16 a, const npyv_u16x3 divisor)
{
    __m256i mulhi = __lasx_xvmuh_hu(a, divisor.val[0]);
    __m256i q = __lasx_xvsub_h(a, mulhi);
            q = __lasx_xvsrl_h(q, divisor.val[1]);
            q = __lasx_xvadd_h(mulhi, q);
            q = __lasx_xvsrl_h(q, divisor.val[2]);
    return  q;
}
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor)
{
    __m256i mulhi = __lasx_xvmuh_h(a, divisor.val[0]);
    __m256i q = __lasx_xvsra_h(__lasx_xvadd_h(a, mulhi), divisor.val[1]);
            q = __lasx_xvsub_h(q, __lasx_xvsrai_h(a, 15));
            q = __lasx_xvsub_h(__lasx_xvxor_v(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
NPY_FINLINE npyv_u32 npyv_divc_u32(npyv_u32 a, const npyv_u32x3 divisor)
{
    __m256i mulhi = __lasx_xvmuh_wu(a, divisor.val[0]);
    __m256i q = __lasx_xvsub_w(a, mulhi);
            q = __lasx_xvsrl_w(q, divisor.val[1]);
            q = __lasx_xvadd_w(mulhi, q);
            q = __lasx_xvsrl_w(q, divisor.val[2]);
    return  q;
}
NPY_FINLINE npyv_s32 npyv_divc_s32(npyv_s32 a, const npyv_s32x3 divisor)
{
    __m256i mulhi = __lasx_xvmuh_w(a, divisor.val[0]);
    __m256i q = __lasx_xvsra_w(__lasx_xvadd_w(a, mulhi), divisor.val[1]);
            q = __lasx_xvsub_w(q, __lasx_xvsrai_w(a, 31));
            q = __lasx_xvsub_w(__lasx_xvxor_v(q, divisor.val[2]), divisor.val[2]);
    return  q;
}
NPY_FINLINE npyv_u64 npyv__mullhi_u64(npyv_u64 a, npyv_u64 b)
{ return __lasx_xvmuh_du(a, b); }
NPY_FINLINE npyv_u64 npyv_divc_u64(npyv_u64 a, const npyv_u64x3 divisor)
{
    __m256i mulhi = __lasx_xvmuh_du(a, divisor.val[0]);
    __m256i q = __lasx_xvsub_d(a, mulhi);
            q = __lasx_xvsrl_d(q, divisor.val[1]);
            q = __lasx_xvadd_d(mulhi, q);
            q = __lasx_xvsrl_d(q, divisor.val[2]);
    return  q;
}
NPY_FINLINE npyv_s64 npyv_divc_s64(npyv_s64 a, const npyv_s64x3 divisor)
{
    __m256i mulhi = __lasx_xvmuh_d(a, divisor.val[0]);
    __m256i q = __lasx_xvsra_d(__lasx_xvadd_d(a, mulhi), divisor.val[1]);
            q = __lasx_xvsub_d(q, __lasx_xvsrai_d(a, 63));
            q = __lasx_xvsub_d(__lasx_xvxor_v(q, divisor.val[2]), divisor.val[2]);
    return  q;
}

/***************************
 * Division
 ***************************/
#define npyv_div_f32  __lasx_xvfdiv_s
#define npyv_div_f64  __lasx_xvfdiv_d

/***************************
 * FUSED
 ***************************/
#define npyv_muladd_f32  __lasx_xvfmadd_s
#define npyv_muladd_f64  __lasx_xvfmadd_d
#define npyv_mulsub_f32  __lasx_xvfmsub_s
#define npyv_mulsub_f64  __lasx_xvfmsub_d
#define npyv_nmuladd_f32 __lasx_xvfnmsub_s
#define npyv_nmuladd_f64 __lasx_xvfnmsub_d
#define npyv_nmulsub_f32 __lasx_xvfnmadd_s
#define npyv_nmulsub_f64 __lasx_xvfnmadd_d

// multiply, add for odd elements and subtract even elements: (a * b) -+ c
NPY_FINLINE npyv_f32 npyv_muladdsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
{
    return __lasx_xvfmadd_s(a, b,
        (__m256)__lasx_xvxor_v((__m256i)c,
            (__m256i)(v8f32){-0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f, -0.0f, 0.0f}));
}
NPY_FINLINE npyv_f64 npyv_muladdsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
{
    return __lasx_xvfmadd_d(a, b,
        (__m256d)__lasx_xvxor_v((__m256i)c,
            (__m256i)(v4f64){-0.0, 0.0, -0.0, 0.0}));
}

/***************************
 * Summation
 ***************************/
// reduce sum across vector
NPY_FINLINE npy_uint32 npyv_sum_u32(npyv_u32 a)
{
    // Horizontal add within each 128-bit half: 4 lanes 32-bit -> 1 lane 128-bit
    __m256i t1 = __lasx_xvhaddw_du_wu(a, a);
    __m256i t2 = __lasx_xvhaddw_qu_du(t1, t1);
    // t2[0..31] holds sum of low half's 4 lanes; t2[128..159] holds sum of high half's 4 lanes
    npy_uint32 lo = (npy_uint32)__lasx_xvpickve2gr_wu(t2, 0);
    npy_uint32 hi = (npy_uint32)__lasx_xvpickve2gr_wu(t2, 4);
    return lo + hi;
}

NPY_FINLINE npy_uint64 npyv_sum_u64(npyv_u64 a)
{
    __m256i t = __lasx_xvhaddw_qu_du(a, a);
    npy_uint64 lo = (npy_uint64)__lasx_xvpickve2gr_du(t, 0);
    npy_uint64 hi = (npy_uint64)__lasx_xvpickve2gr_du(t, 2);
    return lo + hi;
}

NPY_FINLINE float npyv_sum_f32(npyv_f32 a)
{
    // Within each 128-bit half: pairwise add via byte-shift trick
    __m256 t = __lasx_xvfadd_s(a, (__m256)__lasx_xvbsrl_v((__m256i)a, 8));
    t = __lasx_xvfadd_s(t, (__m256)__lasx_xvbsrl_v((__m256i)t, 4));
    // t[0] = sum of first half's 4 lanes, t[4] = sum of second half's 4 lanes
    return t[0] + t[4];
}

NPY_FINLINE double npyv_sum_f64(npyv_f64 a)
{
    // Pairwise add adjacent lanes (within each 128-bit half)
    __m256d t = __lasx_xvfadd_d(a, (__m256d)__lasx_xvbsrl_v((__m256i)a, 8));
    return t[0] + t[2];
}

// expand the source vector and perform sum reduce
NPY_FINLINE npy_uint16 npyv_sumup_u8(npyv_u8 a)
{
    __m256i t1 = __lasx_xvhaddw_hu_bu(a, a);
    __m256i t2 = __lasx_xvhaddw_wu_hu(t1, t1);
    __m256i t3 = __lasx_xvhaddw_du_wu(t2, t2);
    __m256i t4 = __lasx_xvhaddw_qu_du(t3, t3);
    npy_uint32 lo = (npy_uint32)__lasx_xvpickve2gr_wu(t4, 0);
    npy_uint32 hi = (npy_uint32)__lasx_xvpickve2gr_wu(t4, 4);
    return (npy_uint16)(lo + hi);
}

NPY_FINLINE npy_uint32 npyv_sumup_u16(npyv_u16 a)
{
    __m256i t1 = __lasx_xvhaddw_wu_hu(a, a);
    __m256i t2 = __lasx_xvhaddw_du_wu(t1, t1);
    __m256i t3 = __lasx_xvhaddw_qu_du(t2, t2);
    npy_uint32 lo = (npy_uint32)__lasx_xvpickve2gr_wu(t3, 0);
    npy_uint32 hi = (npy_uint32)__lasx_xvpickve2gr_wu(t3, 4);
    return lo + hi;
}

#endif // _NPY_SIMD_LASX_ARITHMETIC_H
