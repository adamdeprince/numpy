#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LASX_REORDER_H
#define _NPY_SIMD_LASX_REORDER_H

/*
 * 256-bit reorder ops.
 * LASX treats most 128-bit-half operations as two independent 128-bit slices,
 * so cross-half work uses xvpermi_q to shuffle the two halves.
 * xvpermi.q semantics for `xvpermi.q xd, xj, imm`:
 *   result low  128 = pick {xj_lo, xj_hi, xd_lo, xd_hi} by imm[1:0]
 *   result high 128 = pick {xj_lo, xj_hi, xd_lo, xd_hi} by imm[5:4]
 *   00 = xj_lo, 01 = xj_hi, 10 = xd_lo, 11 = xd_hi
 */

// combine lower part of two vectors  -> [A_lo, B_lo]
#define npyv_combinel_u8(A, B)  __lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x20)
#define npyv_combinel_s8(A, B)  __lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x20)
#define npyv_combinel_u16(A, B) __lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x20)
#define npyv_combinel_s16(A, B) __lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x20)
#define npyv_combinel_u32(A, B) __lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x20)
#define npyv_combinel_s32(A, B) __lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x20)
#define npyv_combinel_u64(A, B) __lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x20)
#define npyv_combinel_s64(A, B) __lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x20)
#define npyv_combinel_f32(A, B) (__m256)__lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x20)
#define npyv_combinel_f64(A, B) (__m256d)__lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x20)

// combine higher part of two vectors -> [A_hi, B_hi]
#define npyv_combineh_u8(A, B)  __lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x31)
#define npyv_combineh_s8(A, B)  __lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x31)
#define npyv_combineh_u16(A, B) __lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x31)
#define npyv_combineh_s16(A, B) __lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x31)
#define npyv_combineh_u32(A, B) __lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x31)
#define npyv_combineh_s32(A, B) __lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x31)
#define npyv_combineh_u64(A, B) __lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x31)
#define npyv_combineh_s64(A, B) __lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x31)
#define npyv_combineh_f32(A, B) (__m256)__lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x31)
#define npyv_combineh_f64(A, B) (__m256d)__lasx_xvpermi_q((__m256i)B, (__m256i)A, 0x31)

// combine two vectors from lower and higher parts of two other vectors
NPY_FINLINE npyv_s64x2 npyv__combine(__m256i a, __m256i b)
{
    npyv_s64x2 r;
    r.val[0] = npyv_combinel_u8(a, b);
    r.val[1] = npyv_combineh_u8(a, b);
    return r;
}
NPY_FINLINE npyv_f32x2 npyv_combine_f32(__m256 a, __m256 b)
{
    npyv_f32x2 r;
    r.val[0] = npyv_combinel_f32(a, b);
    r.val[1] = npyv_combineh_f32(a, b);
    return r;
}
NPY_FINLINE npyv_f64x2 npyv_combine_f64(__m256d a, __m256d b)
{
    npyv_f64x2 r;
    r.val[0] = npyv_combinel_f64(a, b);
    r.val[1] = npyv_combineh_f64(a, b);
    return r;
}
#define npyv_combine_u8  npyv__combine
#define npyv_combine_s8  npyv__combine
#define npyv_combine_u16 npyv__combine
#define npyv_combine_s16 npyv__combine
#define npyv_combine_u32 npyv__combine
#define npyv_combine_s32 npyv__combine
#define npyv_combine_u64 npyv__combine
#define npyv_combine_s64 npyv__combine

/*
 * interleave (zip) two vectors
 *   xvilvl_b(B, A) within each 128-bit half = interleave-low of (A_half, B_half)
 *   xvilvh_b(B, A) within each 128-bit half = interleave-high of (A_half, B_half)
 *
 * NPYV semantics: r.val[0] is the first nlanes of the interleaved output,
 * r.val[1] is the second nlanes. After xvilvl, the low half of the result
 * is the first 16 bytes of the full interleave; the high half is the third
 * 16 bytes (skipping the second 16). We use xvpermi_q to glue them back into
 * contiguous 32-byte chunks.
 */
#define NPYV_IMPL_LASX_ZIP(T_VEC, SFX, INTR_SFX)                                  \
    NPY_FINLINE T_VEC##x2 npyv_zip_##SFX(T_VEC a, T_VEC b)                        \
    {                                                                             \
        T_VEC##x2 r;                                                              \
        __m256i lo = __lasx_xvilvl_##INTR_SFX((__m256i)b, (__m256i)a);            \
        __m256i hi = __lasx_xvilvh_##INTR_SFX((__m256i)b, (__m256i)a);            \
        r.val[0] = (T_VEC)__lasx_xvpermi_q(hi, lo, 0x20);                         \
        r.val[1] = (T_VEC)__lasx_xvpermi_q(hi, lo, 0x31);                         \
        return r;                                                                 \
    }

NPYV_IMPL_LASX_ZIP(npyv_u8,  u8,  b)
NPYV_IMPL_LASX_ZIP(npyv_s8,  s8,  b)
NPYV_IMPL_LASX_ZIP(npyv_u16, u16, h)
NPYV_IMPL_LASX_ZIP(npyv_s16, s16, h)
NPYV_IMPL_LASX_ZIP(npyv_u32, u32, w)
NPYV_IMPL_LASX_ZIP(npyv_s32, s32, w)
NPYV_IMPL_LASX_ZIP(npyv_u64, u64, d)
NPYV_IMPL_LASX_ZIP(npyv_s64, s64, d)

NPY_FINLINE npyv_f32x2 npyv_zip_f32(__m256 a, __m256 b)
{
    npyv_f32x2 r;
    __m256i lo = __lasx_xvilvl_w((__m256i)b, (__m256i)a);
    __m256i hi = __lasx_xvilvh_w((__m256i)b, (__m256i)a);
    r.val[0] = (__m256)__lasx_xvpermi_q(hi, lo, 0x20);
    r.val[1] = (__m256)__lasx_xvpermi_q(hi, lo, 0x31);
    return r;
}
NPY_FINLINE npyv_f64x2 npyv_zip_f64(__m256d a, __m256d b)
{
    npyv_f64x2 r;
    __m256i lo = __lasx_xvilvl_d((__m256i)b, (__m256i)a);
    __m256i hi = __lasx_xvilvh_d((__m256i)b, (__m256i)a);
    r.val[0] = (__m256d)__lasx_xvpermi_q(hi, lo, 0x20);
    r.val[1] = (__m256d)__lasx_xvpermi_q(hi, lo, 0x31);
    return r;
}

// deinterleave (unzip) two vectors — inverse of zip
#define NPYV_IMPL_LASX_UNZIP(T_VEC, SFX, INTR_SFX)                                \
    NPY_FINLINE T_VEC##x2 npyv_unzip_##SFX(T_VEC a, T_VEC b)                      \
    {                                                                             \
        T_VEC##x2 r;                                                              \
        __m256i ev = __lasx_xvpickev_##INTR_SFX((__m256i)b, (__m256i)a);          \
        __m256i od = __lasx_xvpickod_##INTR_SFX((__m256i)b, (__m256i)a);          \
        r.val[0] = (T_VEC)__lasx_xvpermi_q(ev, ev, 0x00);                         \
        r.val[1] = (T_VEC)__lasx_xvpermi_q(od, od, 0x00);                         \
        /* xvpickev/od work within 128-bit halves, so contiguous reorder needs */ \
        /* a separate gather via xvpermi_q to interleave the halves.            */ \
        r.val[0] = (T_VEC)__lasx_xvpermi_q(ev, ev, 0xd8);                         \
        r.val[1] = (T_VEC)__lasx_xvpermi_q(od, od, 0xd8);                         \
        return r;                                                                 \
    }
/*
 * Note: xvpermi_q imm 0xd8 = 0b11011000:
 *   bits[1:0]=00 -> xd_lo
 *   bits[5:4]=01 -> xd_hi (??? wait that's 0b01 = 1, xd_hi)
 *   so result = [xd_lo, xd_hi] which is identity. Not useful.
 *
 * The actual unzip across 256-bit needs to gather (ev_lo, ev_hi) from two
 * pickev results applied to different operand pairs. For the contiguous case
 * the standard recipe is:
 *   even = pickev(b, a)  -> [a_even_lo, b_even_lo | a_even_hi, b_even_hi]
 *   we want [a_even_lo, a_even_hi, b_even_lo, b_even_hi]
 * which requires a within-vector permute_q. Use 0xd8 = lanes [0,2,1,3].
 * Actually for permi_q on a single source (xj==xd), only 4 lanes [lo, hi] are
 * picked, so a 4-source pattern can't be expressed without a second op.
 *
 * Pragmatic implementation: fall back to scalar via memory. Correct, slow.
 * unzip is not on the exp/log hot path.
 */
#undef NPYV_IMPL_LASX_UNZIP

#define NPYV_IMPL_LASX_UNZIP_SCALAR(T_VEC, SFX, CTYPE)                            \
    NPY_FINLINE T_VEC##x2 npyv_unzip_##SFX(T_VEC a, T_VEC b)                      \
    {                                                                             \
        T_VEC##x2 r;                                                              \
        CTYPE buf_a[npyv_nlanes_##SFX], buf_b[npyv_nlanes_##SFX];                 \
        CTYPE out0[npyv_nlanes_##SFX], out1[npyv_nlanes_##SFX];                   \
        npyv_store_##SFX(buf_a, a);                                               \
        npyv_store_##SFX(buf_b, b);                                               \
        for (int i = 0; i < npyv_nlanes_##SFX/2; i++) {                           \
            out0[i] = buf_a[2*i];                                                 \
            out0[i + npyv_nlanes_##SFX/2] = buf_b[2*i];                           \
            out1[i] = buf_a[2*i + 1];                                             \
            out1[i + npyv_nlanes_##SFX/2] = buf_b[2*i + 1];                       \
        }                                                                         \
        r.val[0] = npyv_load_##SFX(out0);                                         \
        r.val[1] = npyv_load_##SFX(out1);                                         \
        return r;                                                                 \
    }

NPYV_IMPL_LASX_UNZIP_SCALAR(npyv_u8,  u8,  npy_uint8)
NPYV_IMPL_LASX_UNZIP_SCALAR(npyv_s8,  s8,  npy_int8)
NPYV_IMPL_LASX_UNZIP_SCALAR(npyv_u16, u16, npy_uint16)
NPYV_IMPL_LASX_UNZIP_SCALAR(npyv_s16, s16, npy_int16)
NPYV_IMPL_LASX_UNZIP_SCALAR(npyv_u32, u32, npy_uint32)
NPYV_IMPL_LASX_UNZIP_SCALAR(npyv_s32, s32, npy_int32)
NPYV_IMPL_LASX_UNZIP_SCALAR(npyv_u64, u64, npy_uint64)
NPYV_IMPL_LASX_UNZIP_SCALAR(npyv_s64, s64, npy_int64)
NPYV_IMPL_LASX_UNZIP_SCALAR(npyv_f32, f32, float)
NPYV_IMPL_LASX_UNZIP_SCALAR(npyv_f64, f64, double)

// Reverse elements of each 64-bit lane
NPY_FINLINE npyv_u8 npyv_rev64_u8(npyv_u8 a)
{
    v32u8 idx = {7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
                 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8};
    return __lasx_xvshuf_b(a, a, (__m256i)idx);
}
#define npyv_rev64_s8 npyv_rev64_u8

NPY_FINLINE npyv_u16 npyv_rev64_u16(npyv_u16 a)
{
    v16u16 idx = {3, 2, 1, 0, 7, 6, 5, 4, 3, 2, 1, 0, 7, 6, 5, 4};
    return __lasx_xvshuf_h((__m256i)idx, a, a);
}
#define npyv_rev64_s16 npyv_rev64_u16

NPY_FINLINE npyv_u32 npyv_rev64_u32(npyv_u32 a)
{
    v8u32 idx = {1, 0, 3, 2, 1, 0, 3, 2};
    return __lasx_xvshuf_w((__m256i)idx, a, a);
}
#define npyv_rev64_s32 npyv_rev64_u32

NPY_FINLINE npyv_f32 npyv_rev64_f32(npyv_f32 a)
{
    v8i32 idx = {1, 0, 3, 2, 1, 0, 3, 2};
    return (__m256)__lasx_xvshuf_w((__m256i)idx, (__m256i)a, (__m256i)a);
}

// Permuting the elements of each 128-bit lane by immediate index for each element.
// The same pattern E0..E3 (or E0..E1) is applied independently to both halves.
#define npyv_permi128_u32(A, E0, E1, E2, E3)                            \
    npyv_set_u32(                                                       \
       __lasx_xvpickve2gr_wu(A, E0),     __lasx_xvpickve2gr_wu(A, E1),  \
       __lasx_xvpickve2gr_wu(A, E2),     __lasx_xvpickve2gr_wu(A, E3),  \
       __lasx_xvpickve2gr_wu(A, (E0)+4), __lasx_xvpickve2gr_wu(A, (E1)+4), \
       __lasx_xvpickve2gr_wu(A, (E2)+4), __lasx_xvpickve2gr_wu(A, (E3)+4)  \
    )
#define npyv_permi128_s32(A, E0, E1, E2, E3)                            \
    npyv_set_s32(                                                       \
       __lasx_xvpickve2gr_w(A, E0),     __lasx_xvpickve2gr_w(A, E1),    \
       __lasx_xvpickve2gr_w(A, E2),     __lasx_xvpickve2gr_w(A, E3),    \
       __lasx_xvpickve2gr_w(A, (E0)+4), __lasx_xvpickve2gr_w(A, (E1)+4), \
       __lasx_xvpickve2gr_w(A, (E2)+4), __lasx_xvpickve2gr_w(A, (E3)+4)  \
    )
#define npyv_permi128_u64(A, E0, E1)                                    \
    npyv_set_u64(                                                       \
       __lasx_xvpickve2gr_du(A, E0),     __lasx_xvpickve2gr_du(A, E1),  \
       __lasx_xvpickve2gr_du(A, (E0)+2), __lasx_xvpickve2gr_du(A, (E1)+2) \
    )
#define npyv_permi128_s64(A, E0, E1)                                    \
    npyv_set_s64(                                                       \
       __lasx_xvpickve2gr_d(A, E0),     __lasx_xvpickve2gr_d(A, E1),    \
       __lasx_xvpickve2gr_d(A, (E0)+2), __lasx_xvpickve2gr_d(A, (E1)+2) \
    )
#define npyv_permi128_f32(A, E0, E1, E2, E3)                            \
    (__m256)__lasx_xvshuf_w((__m256i)(v8u32){E0, E1, E2, E3, E0, E1, E2, E3}, \
                            (__m256i)A, (__m256i)A)
#define npyv_permi128_f64(A, E0, E1)                                    \
    (__m256d)__lasx_xvshuf_d((__m256i)(v4i64){E0, E1, E0, E1},          \
                             (__m256i)A, (__m256i)A)

#endif // _NPY_SIMD_LASX_REORDER_H
