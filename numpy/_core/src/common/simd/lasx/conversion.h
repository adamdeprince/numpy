#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LASX_CVT_H
#define _NPY_SIMD_LASX_CVT_H

// convert mask types to integer types
#define npyv_cvt_u8_b8(BL)   BL
#define npyv_cvt_s8_b8(BL)   BL
#define npyv_cvt_u16_b16(BL) BL
#define npyv_cvt_s16_b16(BL) BL
#define npyv_cvt_u32_b32(BL) BL
#define npyv_cvt_s32_b32(BL) BL
#define npyv_cvt_u64_b64(BL) BL
#define npyv_cvt_s64_b64(BL) BL
#define npyv_cvt_f32_b32(BL) (__m256)(BL)
#define npyv_cvt_f64_b64(BL) (__m256d)(BL)

// convert integer types to mask types
#define npyv_cvt_b8_u8(A)   A
#define npyv_cvt_b8_s8(A)   A
#define npyv_cvt_b16_u16(A) A
#define npyv_cvt_b16_s16(A) A
#define npyv_cvt_b32_u32(A) A
#define npyv_cvt_b32_s32(A) A
#define npyv_cvt_b64_u64(A) A
#define npyv_cvt_b64_s64(A) A
#define npyv_cvt_b32_f32(A) (__m256i)(A)
#define npyv_cvt_b64_f64(A) (__m256i)(A)

// convert boolean vector to integer bitfield
// xvmsknz_b stores a 16-bit byte-nonzero mask per 128-bit half (lane 0 of each)
// We combine both halves into a 32-bit integer.
NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a)
{
    __m256i m = __lasx_xvmsknz_b(a);
    npy_uint32 lo = (npy_uint32)__lasx_xvpickve2gr_wu(m, 0) & 0xFFFFu;
    npy_uint32 hi = (npy_uint32)__lasx_xvpickve2gr_wu(m, 4) & 0xFFFFu;
    return (npy_uint64)(lo | (hi << 16));
}
NPY_FINLINE npy_uint64 npyv_tobits_b16(npyv_b16 a)
{
    // Each 16-bit b16 lane is all-1s (0xFFFF, sign bit set) or all-0s.
    // xvmskltz_h packs the sign bit of each 16-bit lane into a per-half
    // 8-bit mask (8 lanes per 128-bit half).  Combine the two halves to
    // get a 16-bit mask covering all 16 lanes.  The previous pickev_b
    // version only saw the low 8 lanes.
    __m256i m = __lasx_xvmskltz_h(a);
    npy_uint32 lo = (npy_uint32)__lasx_xvpickve2gr_w(m, 0) & 0xFFu;
    npy_uint32 hi = (npy_uint32)__lasx_xvpickve2gr_w(m, 4) & 0xFFu;
    return (npy_uint64)(lo | (hi << 8));
}
NPY_FINLINE npy_uint64 npyv_tobits_b32(npyv_b32 a)
{
    __m256i m = __lasx_xvmskltz_w(a);
    npy_uint32 lo = (npy_uint32)__lasx_xvpickve2gr_w(m, 0);
    npy_uint32 hi = (npy_uint32)__lasx_xvpickve2gr_w(m, 4);
    return (npy_uint64)(lo | (hi << 4));  // 4 lanes per half
}
NPY_FINLINE npy_uint64 npyv_tobits_b64(npyv_b64 a)
{
    __m256i m = __lasx_xvmskltz_d(a);
    npy_uint64 lo = (npy_uint64)__lasx_xvpickve2gr_d(m, 0);
    npy_uint64 hi = (npy_uint64)__lasx_xvpickve2gr_d(m, 2);
    return lo | (hi << 2);  // 2 lanes per half
}

// expand 8-bit unsigned -> two vectors of 16-bit unsigned
// xvsllwil expands within each 128-bit half (low 8 bytes -> 8 u16s in that half)
// xvexth expands within each 128-bit half (high 8 bytes -> 8 u16s)
// We then use xvpermi_q to reassemble contiguous halves.
NPY_FINLINE npyv_u16x2 npyv_expand_u16_u8(npyv_u8 data) {
    npyv_u16x2 r;
    __m256i lo_ext = __lasx_xvsllwil_hu_bu(data, 0); // [u16s of bytes 0-7 | u16s of bytes 16-23]
    __m256i hi_ext = __lasx_xvexth_hu_bu(data);     // [u16s of bytes 8-15 | u16s of bytes 24-31]
    // val[0] wants u16s of bytes 0-15  = [lo_ext.low, hi_ext.low]
    // val[1] wants u16s of bytes 16-31 = [lo_ext.high, hi_ext.high]
    r.val[0] = __lasx_xvpermi_q(lo_ext, hi_ext, 0x20);
    r.val[1] = __lasx_xvpermi_q(lo_ext, hi_ext, 0x31);
    return r;
}

NPY_FINLINE npyv_u32x2 npyv_expand_u32_u16(npyv_u16 data) {
    npyv_u32x2 r;
    __m256i lo_ext = __lasx_xvsllwil_wu_hu(data, 0);
    __m256i hi_ext = __lasx_xvexth_wu_hu(data);
    r.val[0] = __lasx_xvpermi_q(lo_ext, hi_ext, 0x20);
    r.val[1] = __lasx_xvpermi_q(lo_ext, hi_ext, 0x31);
    return r;
}

// pack two 16-bit boolean into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b16(npyv_b16 a, npyv_b16 b) {
    __m256i sb = __lasx_xvsat_h(b, 7);
    __m256i sa = __lasx_xvsat_h(a, 7);
    __m256i packed = __lasx_xvpickev_b(sb, sa);
    // packed has [a_low_packed, b_low_packed | a_high_packed, b_high_packed]
    // We want [a_packed_all, b_packed_all] = swap middle halves
    return __lasx_xvpermi_d(packed, 0xd8);  // d8 = 0b11011000 = [0,2,1,3] in 64-bit lanes
}

// pack four 32-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d) {
    __m256i ab_h = __lasx_xvpickev_h(__lasx_xvsat_w(b, 15), __lasx_xvsat_w(a, 15));
    __m256i cd_h = __lasx_xvpickev_h(__lasx_xvsat_w(d, 15), __lasx_xvsat_w(c, 15));
    ab_h = __lasx_xvpermi_d(ab_h, 0xd8);
    cd_h = __lasx_xvpermi_d(cd_h, 0xd8);
    return npyv_pack_b8_b16(ab_h, cd_h);
}

// pack eight 64-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                 npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h) {
    __m256i ab = __lasx_xvpermi_d(
        __lasx_xvpickev_h(__lasx_xvsat_w(b, 15), __lasx_xvsat_w(a, 15)), 0xd8);
    __m256i cd = __lasx_xvpermi_d(
        __lasx_xvpickev_h(__lasx_xvsat_w(d, 15), __lasx_xvsat_w(c, 15)), 0xd8);
    __m256i ef = __lasx_xvpermi_d(
        __lasx_xvpickev_h(__lasx_xvsat_w(f, 15), __lasx_xvsat_w(e, 15)), 0xd8);
    __m256i gh = __lasx_xvpermi_d(
        __lasx_xvpickev_h(__lasx_xvsat_w(h, 15), __lasx_xvsat_w(g, 15)), 0xd8);
    return npyv_pack_b8_b32(ab, cd, ef, gh);
}

// round to nearest integer (even)
#define npyv_round_s32_f32 __lasx_xvftintrne_w_s
NPY_FINLINE npyv_s32 npyv_round_s32_f64(npyv_f64 a, npyv_f64 b)
{
    // xvftintrne_w_d packs two f64 vectors (4+4 doubles) into one s32 (8 lanes)
    // The intrinsic operates within 128-bit halves, producing:
    // [a[0]i, a[1]i, b[0]i, b[1]i | a[2]i, a[3]i, b[2]i, b[3]i]
    // We want contiguous [a[0..3]i, b[0..3]i], so permute 32-bit lanes.
    __m256i raw = __lasx_xvftintrne_w_d(b, a);
    return __lasx_xvshuf_w((__m256i)(v8i32){0, 1, 4, 5, 2, 3, 6, 7}, raw, raw);
}

#endif // _NPY_SIMD_LASX_CVT_H
