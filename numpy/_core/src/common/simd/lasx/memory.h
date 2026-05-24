#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_LASX_MEMORY_H
#define _NPY_SIMD_LASX_MEMORY_H

#include <stdint.h>
#include "misc.h"

/***************************
 * load/store
 ***************************/
#define NPYV_IMPL_LASX_MEM(SFX, CTYPE)                              \
    NPY_FINLINE npyv_##SFX npyv_load_##SFX(const CTYPE *ptr)        \
    { return (npyv_##SFX)(__lasx_xvld(ptr, 0)); }                   \
    NPY_FINLINE npyv_##SFX npyv_loada_##SFX(const CTYPE *ptr)       \
    { return (npyv_##SFX)(__lasx_xvld(ptr, 0)); }                   \
    NPY_FINLINE npyv_##SFX npyv_loads_##SFX(const CTYPE *ptr)       \
    { return (npyv_##SFX)(__lasx_xvld(ptr, 0)); }                   \
    NPY_FINLINE npyv_##SFX npyv_loadl_##SFX(const CTYPE *ptr)       \
    { return (npyv_##SFX)__lasx_xvldrepl_d(ptr, 0); }               \
    NPY_FINLINE void npyv_store_##SFX(CTYPE *ptr, npyv_##SFX vec)   \
    { __lasx_xvst(vec, ptr, 0); }                                   \
    NPY_FINLINE void npyv_storea_##SFX(CTYPE *ptr, npyv_##SFX vec)  \
    { __lasx_xvst(vec, ptr, 0); }                                   \
    NPY_FINLINE void npyv_stores_##SFX(CTYPE *ptr, npyv_##SFX vec)  \
    { __lasx_xvst(vec, ptr, 0); }                                   \
    NPY_FINLINE void npyv_storel_##SFX(CTYPE *ptr, npyv_##SFX vec)  \
    { __lasx_xvstelm_d(vec, ptr, 0, 0); }                           \
    NPY_FINLINE void npyv_storeh_##SFX(CTYPE *ptr, npyv_##SFX vec)  \
    { __lasx_xvstelm_d(vec, ptr, 0, 1); }

NPYV_IMPL_LASX_MEM(u8,  npy_uint8)
NPYV_IMPL_LASX_MEM(s8,  npy_int8)
NPYV_IMPL_LASX_MEM(u16, npy_uint16)
NPYV_IMPL_LASX_MEM(s16, npy_int16)
NPYV_IMPL_LASX_MEM(u32, npy_uint32)
NPYV_IMPL_LASX_MEM(s32, npy_int32)
NPYV_IMPL_LASX_MEM(u64, npy_uint64)
NPYV_IMPL_LASX_MEM(s64, npy_int64)
NPYV_IMPL_LASX_MEM(f32, float)
NPYV_IMPL_LASX_MEM(f64, double)

/***************************
 * Non-contiguous Load
 ***************************/
//// 32 (8 lanes)
NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, npy_intp stride)
{
    __m256i a = __lasx_xvreplgr2vr_w(ptr[0]);
    a = __lasx_xvinsgr2vr_w(a, ptr[stride],   1);
    a = __lasx_xvinsgr2vr_w(a, ptr[stride*2], 2);
    a = __lasx_xvinsgr2vr_w(a, ptr[stride*3], 3);
    a = __lasx_xvinsgr2vr_w(a, ptr[stride*4], 4);
    a = __lasx_xvinsgr2vr_w(a, ptr[stride*5], 5);
    a = __lasx_xvinsgr2vr_w(a, ptr[stride*6], 6);
    a = __lasx_xvinsgr2vr_w(a, ptr[stride*7], 7);
    return a;
}
NPY_FINLINE npyv_u32 npyv_loadn_u32(const npy_uint32 *ptr, npy_intp stride)
{ return npyv_reinterpret_u32_s32(npyv_loadn_s32((const npy_int32*)ptr, stride)); }
NPY_FINLINE npyv_f32 npyv_loadn_f32(const float *ptr, npy_intp stride)
{ return npyv_reinterpret_f32_s32(npyv_loadn_s32((const npy_int32*)ptr, stride)); }

//// 64 (4 lanes)
NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, npy_intp stride)
{
    __m256i a = __lasx_xvreplgr2vr_d(ptr[0]);
    a = __lasx_xvinsgr2vr_d(a, ptr[stride],   1);
    a = __lasx_xvinsgr2vr_d(a, ptr[stride*2], 2);
    a = __lasx_xvinsgr2vr_d(a, ptr[stride*3], 3);
    return a;
}
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, npy_intp stride)
{ return npyv_reinterpret_f64_s64(npyv_loadn_s64((const npy_int64*)ptr, stride)); }
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, npy_intp stride)
{ return npyv_reinterpret_u64_s64(npyv_loadn_s64((const npy_int64*)ptr, stride)); }

//// 64-bit load over 32-bit stride (4 pairs of 32-bit, gathered with 64-bit stride between pairs)
NPY_FINLINE npyv_s32 npyv_loadn2_s32(const npy_int32 *ptr, npy_intp stride)
{
    __m256i a = npyv_setall_s64(0);
    a = __lasx_xvinsgr2vr_d(a, ((const long long*)ptr)[0],            0);
    a = __lasx_xvinsgr2vr_d(a, *(const long long*)(ptr + stride),     1);
    a = __lasx_xvinsgr2vr_d(a, *(const long long*)(ptr + stride*2),   2);
    a = __lasx_xvinsgr2vr_d(a, *(const long long*)(ptr + stride*3),   3);
    return a;
}
NPY_FINLINE npyv_u32 npyv_loadn2_u32(const npy_uint32 *ptr, npy_intp stride)
{ return npyv_reinterpret_u32_s32(npyv_loadn2_s32((const npy_int32*)ptr, stride)); }
NPY_FINLINE npyv_f32 npyv_loadn2_f32(const float *ptr, npy_intp stride)
{ return npyv_reinterpret_f32_s32(npyv_loadn2_s32((const npy_int32*)ptr, stride)); }

//// 128-bit load over 64-bit stride (2 pairs of 64-bit, gathered with 128-bit stride between pairs)
NPY_FINLINE npyv_s64 npyv_loadn2_s64(const npy_int64 *ptr, npy_intp stride)
{
    __m256i a = npyv_setall_s64(0);
    a = __lasx_xvinsgr2vr_d(a, ptr[0],           0);
    a = __lasx_xvinsgr2vr_d(a, ptr[1],           1);
    a = __lasx_xvinsgr2vr_d(a, ptr[stride],      2);
    a = __lasx_xvinsgr2vr_d(a, ptr[stride + 1],  3);
    return a;
}
NPY_FINLINE npyv_u64 npyv_loadn2_u64(const npy_uint64 *ptr, npy_intp stride)
{ return npyv_reinterpret_u64_s64(npyv_loadn2_s64((const npy_int64*)ptr, stride)); }
NPY_FINLINE npyv_f64 npyv_loadn2_f64(const double *ptr, npy_intp stride)
{ return npyv_reinterpret_f64_s64(npyv_loadn2_s64((const npy_int64*)ptr, stride)); }

/***************************
 * Non-contiguous Store
 ***************************/
//// 32 (8 lanes)
NPY_FINLINE void npyv_storen_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{
    __lasx_xvstelm_w(a, ptr,            0, 0);
    __lasx_xvstelm_w(a, ptr + stride,   0, 1);
    __lasx_xvstelm_w(a, ptr + stride*2, 0, 2);
    __lasx_xvstelm_w(a, ptr + stride*3, 0, 3);
    __lasx_xvstelm_w(a, ptr + stride*4, 0, 4);
    __lasx_xvstelm_w(a, ptr + stride*5, 0, 5);
    __lasx_xvstelm_w(a, ptr + stride*6, 0, 6);
    __lasx_xvstelm_w(a, ptr + stride*7, 0, 7);
}
NPY_FINLINE void npyv_storen_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{ npyv_storen_s32((npy_int32*)ptr, stride, a); }
NPY_FINLINE void npyv_storen_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen_s32((npy_int32*)ptr, stride, (npyv_s32)a); }

//// 64 (4 lanes)
NPY_FINLINE void npyv_storen_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{
    __lasx_xvstelm_d(a, ptr,            0, 0);
    __lasx_xvstelm_d(a, ptr + stride,   0, 1);
    __lasx_xvstelm_d(a, ptr + stride*2, 0, 2);
    __lasx_xvstelm_d(a, ptr + stride*3, 0, 3);
}
NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{ npyv_storen_s64((npy_int64*)ptr, stride, (npyv_s64)a); }
NPY_FINLINE void npyv_storen_f64(double *ptr, npy_intp stride, npyv_f64 a)
{ npyv_storen_s64((npy_int64*)ptr, stride, (npyv_s64)a); }

//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{
    __lasx_xvstelm_d((__m256i)a, ptr,            0, 0);
    __lasx_xvstelm_d((__m256i)a, ptr + stride,   0, 1);
    __lasx_xvstelm_d((__m256i)a, ptr + stride*2, 0, 2);
    __lasx_xvstelm_d((__m256i)a, ptr + stride*3, 0, 3);
}
NPY_FINLINE void npyv_storen2_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, a); }
NPY_FINLINE void npyv_storen2_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, (npyv_u32)a); }

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{
    ptr[0]          = __lasx_xvpickve2gr_d(a, 0);
    ptr[1]          = __lasx_xvpickve2gr_d(a, 1);
    ptr[stride]     = __lasx_xvpickve2gr_d(a, 2);
    ptr[stride + 1] = __lasx_xvpickve2gr_d(a, 3);
}
NPY_FINLINE void npyv_storen2_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{ npyv_storen2_s64((npy_int64*)ptr, stride, (npyv_s64)a); }
NPY_FINLINE void npyv_storen2_f64(double *ptr, npy_intp stride, npyv_f64 a)
{ npyv_storen2_s64((npy_int64*)ptr, stride, (npyv_s64)a); }

/*********************************
 * Partial Load
 *********************************/
//// 32 (8 lanes)
NPY_FINLINE npyv_s32 npyv_load_till_s32(const npy_int32 *ptr, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0);
    if (nlane >= 8) return npyv_load_s32(ptr);
    __m256i vfill = npyv_setall_s32(fill);
    switch(nlane) {
        case 7: vfill = __lasx_xvinsgr2vr_w(vfill, ptr[6], 6); /* fallthrough */
        case 6: vfill = __lasx_xvinsgr2vr_w(vfill, ptr[5], 5); /* fallthrough */
        case 5: vfill = __lasx_xvinsgr2vr_w(vfill, ptr[4], 4); /* fallthrough */
        case 4: vfill = __lasx_xvinsgr2vr_w(vfill, ptr[3], 3); /* fallthrough */
        case 3: vfill = __lasx_xvinsgr2vr_w(vfill, ptr[2], 2); /* fallthrough */
        case 2: vfill = __lasx_xvinsgr2vr_w(vfill, ptr[1], 1); /* fallthrough */
        case 1: vfill = __lasx_xvinsgr2vr_w(vfill, ptr[0], 0); break;
    }
    return vfill;
}
NPY_FINLINE npyv_s32 npyv_load_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return npyv_load_till_s32(ptr, nlane, 0); }

//// 64 (4 lanes)
NPY_FINLINE npyv_s64 npyv_load_till_s64(const npy_int64 *ptr, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    if (nlane >= 4) return npyv_load_s64(ptr);
    __m256i vfill = npyv_setall_s64(fill);
    switch(nlane) {
        case 3: vfill = __lasx_xvinsgr2vr_d(vfill, ptr[2], 2); /* fallthrough */
        case 2: vfill = __lasx_xvinsgr2vr_d(vfill, ptr[1], 1); /* fallthrough */
        case 1: vfill = __lasx_xvinsgr2vr_d(vfill, ptr[0], 0); break;
    }
    return vfill;
}
NPY_FINLINE npyv_s64 npyv_load_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{ return npyv_load_till_s64(ptr, nlane, 0); }

//// 64-bit nlane (load pairs of 32-bit; up to 4 pairs)
NPY_FINLINE npyv_s32 npyv_load2_till_s32(const npy_int32 *ptr, npy_uintp nlane,
                                         npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    if (nlane >= 4) return npyv_load_s32(ptr);
    const __m256i vfill = npyv_set_s32(fill_lo, fill_hi, fill_lo, fill_hi,
                                       fill_lo, fill_hi, fill_lo, fill_hi);
    __m256i r = vfill;
    switch(nlane) {
        case 3: r = __lasx_xvinsgr2vr_d(r, ((const long long*)ptr)[2], 2); /* fallthrough */
        case 2: r = __lasx_xvinsgr2vr_d(r, ((const long long*)ptr)[1], 1); /* fallthrough */
        case 1: r = __lasx_xvinsgr2vr_d(r, ((const long long*)ptr)[0], 0); break;
    }
    return r;
}
NPY_FINLINE npyv_s32 npyv_load2_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return (npyv_s32)npyv_load_tillz_s64((const npy_int64*)ptr, nlane); }

//// 128-bit nlane (load pairs of 64-bit; up to 2 pairs)
NPY_FINLINE npyv_s64 npyv_load2_till_s64(const npy_int64 *ptr, npy_uintp nlane,
                                         npy_int64 fill_lo, npy_int64 fill_hi)
{
    assert(nlane > 0);
    if (nlane >= 2) return npyv_load_s64(ptr);
    const __m256i vfill = npyv_set_s64(fill_lo, fill_hi, fill_lo, fill_hi);
    __m256i r = __lasx_xvinsgr2vr_d(vfill, ptr[0], 0);
    r = __lasx_xvinsgr2vr_d(r, ptr[1], 1);
    return r;
}
NPY_FINLINE npyv_s64 npyv_load2_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
    assert(nlane > 0);
    if (nlane >= 2) return npyv_load_s64(ptr);
    __m256i r = npyv_setall_s64(0);
    r = __lasx_xvinsgr2vr_d(r, ptr[0], 0);
    r = __lasx_xvinsgr2vr_d(r, ptr[1], 1);
    return r;
}

/*********************************
 * Non-contiguous partial load
 *********************************/
//// 32 (8 lanes)
NPY_FINLINE npyv_s32
npyv_loadn_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0);
    if (nlane >= 8) return npyv_loadn_s32(ptr, stride);
    __m256i vfill = npyv_setall_s32(fill);
    switch(nlane) {
        case 7: vfill = __lasx_xvinsgr2vr_w(vfill, ptr[stride*6], 6); /* fallthrough */
        case 6: vfill = __lasx_xvinsgr2vr_w(vfill, ptr[stride*5], 5); /* fallthrough */
        case 5: vfill = __lasx_xvinsgr2vr_w(vfill, ptr[stride*4], 4); /* fallthrough */
        case 4: vfill = __lasx_xvinsgr2vr_w(vfill, ptr[stride*3], 3); /* fallthrough */
        case 3: vfill = __lasx_xvinsgr2vr_w(vfill, ptr[stride*2], 2); /* fallthrough */
        case 2: vfill = __lasx_xvinsgr2vr_w(vfill, ptr[stride*1], 1); /* fallthrough */
        case 1: vfill = __lasx_xvinsgr2vr_w(vfill, ptr[0],        0); break;
    }
    return vfill;
}
NPY_FINLINE npyv_s32 npyv_loadn_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn_till_s32(ptr, stride, nlane, 0); }

//// 64 (4 lanes)
NPY_FINLINE npyv_s64
npyv_loadn_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    if (nlane >= 4) return npyv_loadn_s64(ptr, stride);
    __m256i vfill = npyv_setall_s64(fill);
    switch(nlane) {
        case 3: vfill = __lasx_xvinsgr2vr_d(vfill, ptr[stride*2], 2); /* fallthrough */
        case 2: vfill = __lasx_xvinsgr2vr_d(vfill, ptr[stride*1], 1); /* fallthrough */
        case 1: vfill = __lasx_xvinsgr2vr_d(vfill, ptr[0],        0); break;
    }
    return vfill;
}
NPY_FINLINE npyv_s64 npyv_loadn_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn_till_s64(ptr, stride, nlane, 0); }

//// 64-bit load over 32-bit stride (up to 4 pairs)
NPY_FINLINE npyv_s32 npyv_loadn2_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane,
                                          npy_int32 fill_lo, npy_int32 fill_hi)
{
    assert(nlane > 0);
    if (nlane >= 4) return npyv_loadn2_s32(ptr, stride);
    const __m256i vfill = npyv_set_s32(fill_lo, fill_hi, fill_lo, fill_hi,
                                       fill_lo, fill_hi, fill_lo, fill_hi);
    __m256i r = vfill;
    switch(nlane) {
        case 3: r = __lasx_xvinsgr2vr_d(r, *(const long long*)(ptr + stride*2), 2); /* fallthrough */
        case 2: r = __lasx_xvinsgr2vr_d(r, *(const long long*)(ptr + stride*1), 1); /* fallthrough */
        case 1: r = __lasx_xvinsgr2vr_d(r, *(const long long*)(ptr),            0); break;
    }
    return r;
}
NPY_FINLINE npyv_s32 npyv_loadn2_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn2_till_s32(ptr, stride, nlane, 0, 0); }

//// 128-bit load over 64-bit stride (up to 2 pairs)
NPY_FINLINE npyv_s64 npyv_loadn2_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane,
                                          npy_int64 fill_lo, npy_int64 fill_hi)
{
    assert(nlane > 0);
    if (nlane >= 2) return npyv_loadn2_s64(ptr, stride);
    const __m256i vfill = npyv_set_s64(fill_lo, fill_hi, fill_lo, fill_hi);
    __m256i r = __lasx_xvinsgr2vr_d(vfill, ptr[0], 0);
    r = __lasx_xvinsgr2vr_d(r, ptr[1], 1);
    return r;
}
NPY_FINLINE npyv_s64 npyv_loadn2_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{
    assert(nlane > 0);
    if (nlane >= 2) return npyv_loadn2_s64(ptr, stride);
    __m256i r = npyv_setall_s64(0);
    r = __lasx_xvinsgr2vr_d(r, ptr[0], 0);
    r = __lasx_xvinsgr2vr_d(r, ptr[1], 1);
    return r;
}

/*********************************
 * Partial store
 *********************************/
//// 32 (8 lanes)
NPY_FINLINE void npyv_store_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    if (nlane >= 8) { npyv_store_s32(ptr, a); return; }
    switch(nlane) {
        case 7: __lasx_xvstelm_w(a, ptr + 6, 0, 6); /* fallthrough */
        case 6: __lasx_xvstelm_w(a, ptr + 5, 0, 5); /* fallthrough */
        case 5: __lasx_xvstelm_w(a, ptr + 4, 0, 4); /* fallthrough */
        case 4: __lasx_xvstelm_w(a, ptr + 3, 0, 3); /* fallthrough */
        case 3: __lasx_xvstelm_w(a, ptr + 2, 0, 2); /* fallthrough */
        case 2: __lasx_xvstelm_w(a, ptr + 1, 0, 1); /* fallthrough */
        case 1: __lasx_xvstelm_w(a, ptr + 0, 0, 0); break;
    }
}
//// 64 (4 lanes)
NPY_FINLINE void npyv_store_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    if (nlane >= 4) { npyv_store_s64(ptr, a); return; }
    switch(nlane) {
        case 3: __lasx_xvstelm_d(a, ptr + 2, 0, 2); /* fallthrough */
        case 2: __lasx_xvstelm_d(a, ptr + 1, 0, 1); /* fallthrough */
        case 1: __lasx_xvstelm_d(a, ptr + 0, 0, 0); break;
    }
}
//// 64-bit nlane (up to 4 pairs of 32-bit)
NPY_FINLINE void npyv_store2_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{ npyv_store_till_s64((npy_int64*)ptr, nlane, (npyv_s64)a); }

//// 128-bit nlane (up to 2 pairs of 64-bit)
NPY_FINLINE void npyv_store2_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    if (nlane >= 2) { npyv_store_s64(ptr, a); return; }
    __lasx_xvstelm_d(a, ptr + 0, 0, 0);
    __lasx_xvstelm_d(a, ptr + 1, 0, 1);
}

/*********************************
 * Non-contiguous partial store
 *********************************/
//// 32 (8 lanes)
NPY_FINLINE void npyv_storen_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    if (nlane >= 8) { npyv_storen_s32(ptr, stride, a); return; }
    __lasx_xvstelm_w(a, ptr, 0, 0);
    switch(nlane) {
        case 1: return;
        case 7: ptr[stride*6] = __lasx_xvpickve2gr_w(a, 6); /* fallthrough */
        case 6: ptr[stride*5] = __lasx_xvpickve2gr_w(a, 5); /* fallthrough */
        case 5: ptr[stride*4] = __lasx_xvpickve2gr_w(a, 4); /* fallthrough */
        case 4: ptr[stride*3] = __lasx_xvpickve2gr_w(a, 3); /* fallthrough */
        case 3: ptr[stride*2] = __lasx_xvpickve2gr_w(a, 2); /* fallthrough */
        case 2: ptr[stride*1] = __lasx_xvpickve2gr_w(a, 1); break;
    }
}
//// 64 (4 lanes)
NPY_FINLINE void npyv_storen_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    if (nlane >= 4) { npyv_storen_s64(ptr, stride, a); return; }
    __lasx_xvstelm_d(a, ptr, 0, 0);
    switch(nlane) {
        case 1: return;
        case 3: ptr[stride*2] = __lasx_xvpickve2gr_d(a, 2); /* fallthrough */
        case 2: ptr[stride*1] = __lasx_xvpickve2gr_d(a, 1); break;
    }
}

//// 64-bit store over 32-bit stride (up to 4 pairs)
NPY_FINLINE void npyv_storen2_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    if (nlane >= 4) { npyv_storen2_s32(ptr, stride, a); return; }
    __lasx_xvstelm_d((__m256i)a, ptr, 0, 0);
    switch(nlane) {
        case 1: return;
        case 3: __lasx_xvstelm_d((__m256i)a, ptr + stride*2, 0, 2); /* fallthrough */
        case 2: __lasx_xvstelm_d((__m256i)a, ptr + stride*1, 0, 1); break;
    }
}

//// 128-bit store over 64-bit stride (up to 2 pairs)
NPY_FINLINE void npyv_storen2_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    if (nlane >= 2) { npyv_storen2_s64(ptr, stride, a); return; }
    ptr[0] = __lasx_xvpickve2gr_d(a, 0);
    ptr[1] = __lasx_xvpickve2gr_d(a, 1);
}

/*****************************************************************
 * Implement partial load/store for u32/f32/u64/f64... via casting
 *****************************************************************/
#define NPYV_IMPL_LASX_REST_PARTIAL_TYPES(F_SFX, T_SFX)                                     \
    NPY_FINLINE npyv_##F_SFX npyv_load_till_##F_SFX                                         \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_lanetype_##F_SFX fill)         \
    {                                                                                       \
        union {                                                                             \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        } pun;                                                                              \
        pun.from_##F_SFX = fill;                                                            \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_till_##T_SFX(                   \
            (const npyv_lanetype_##T_SFX *)ptr, nlane, pun.to_##T_SFX                       \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_loadn_till_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane,                    \
     npyv_lanetype_##F_SFX fill)                                                            \
    {                                                                                       \
        union {                                                                             \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        } pun;                                                                              \
        pun.from_##F_SFX = fill;                                                            \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_till_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane, pun.to_##T_SFX               \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_load_tillz_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                                     \
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_tillz_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, nlane                                       \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_loadn_tillz_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane)                    \
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_tillz_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane                               \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE void npyv_store_till_##F_SFX                                                \
    (npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_##F_SFX a)                           \
    {                                                                                       \
        npyv_store_till_##T_SFX(                                                            \
            (npyv_lanetype_##T_SFX *)ptr, nlane,                                            \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }                                                                                       \
    NPY_FINLINE void npyv_storen_till_##F_SFX                                               \
    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)          \
    {                                                                                       \
        npyv_storen_till_##T_SFX(                                                           \
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,                                    \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }

NPYV_IMPL_LASX_REST_PARTIAL_TYPES(u32, s32)
NPYV_IMPL_LASX_REST_PARTIAL_TYPES(f32, s32)
NPYV_IMPL_LASX_REST_PARTIAL_TYPES(u64, s64)
NPYV_IMPL_LASX_REST_PARTIAL_TYPES(f64, s64)

// 128-bit/64-bit stride
#define NPYV_IMPL_LASX_REST_PARTIAL_TYPES_PAIR(F_SFX, T_SFX)                                \
    NPY_FINLINE npyv_##F_SFX npyv_load2_till_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane,                                     \
     npyv_lanetype_##F_SFX fill_lo, npyv_lanetype_##F_SFX fill_hi)                          \
    {                                                                                       \
        union pun {                                                                         \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        };                                                                                  \
        union pun pun_lo;                                                                   \
        union pun pun_hi;                                                                   \
        pun_lo.from_##F_SFX = fill_lo;                                                      \
        pun_hi.from_##F_SFX = fill_hi;                                                      \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load2_till_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, nlane, pun_lo.to_##T_SFX, pun_hi.to_##T_SFX \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_loadn2_till_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane,                    \
     npyv_lanetype_##F_SFX fill_lo, npyv_lanetype_##F_SFX fill_hi)                          \
    {                                                                                       \
        union pun {                                                                         \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        };                                                                                  \
        union pun pun_lo;                                                                   \
        union pun pun_hi;                                                                   \
        pun_lo.from_##F_SFX = fill_lo;                                                      \
        pun_hi.from_##F_SFX = fill_hi;                                                      \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn2_till_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane, pun_lo.to_##T_SFX,           \
            pun_hi.to_##T_SFX                                                               \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_load2_tillz_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                                     \
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load2_tillz_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, nlane                                       \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_loadn2_tillz_##F_SFX                                      \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane)                    \
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn2_tillz_##T_SFX(                \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane                               \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE void npyv_store2_till_##F_SFX                                               \
    (npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_##F_SFX a)                           \
    {                                                                                       \
        npyv_store2_till_##T_SFX(                                                           \
            (npyv_lanetype_##T_SFX *)ptr, nlane,                                            \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }                                                                                       \
    NPY_FINLINE void npyv_storen2_till_##F_SFX                                              \
    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)          \
    {                                                                                       \
        npyv_storen2_till_##T_SFX(                                                          \
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,                                    \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }

NPYV_IMPL_LASX_REST_PARTIAL_TYPES_PAIR(u32, s32)
NPYV_IMPL_LASX_REST_PARTIAL_TYPES_PAIR(f32, s32)
NPYV_IMPL_LASX_REST_PARTIAL_TYPES_PAIR(u64, s64)
NPYV_IMPL_LASX_REST_PARTIAL_TYPES_PAIR(f64, s64)

/************************************************************
 *  de-interleave load / interleave contiguous store
 ************************************************************/
#define NPYV_IMPL_LASX_MEM_INTERLEAVE(SFX, ZSFX)                             \
    NPY_FINLINE npyv_##SFX##x2 npyv_zip_##SFX(npyv_##SFX, npyv_##SFX);       \
    NPY_FINLINE npyv_##SFX##x2 npyv_unzip_##SFX(npyv_##SFX, npyv_##SFX);     \
    NPY_FINLINE npyv_##SFX##x2 npyv_load_##SFX##x2(                          \
        const npyv_lanetype_##SFX *ptr                                       \
    ) {                                                                      \
        return npyv_unzip_##SFX(                                             \
         npyv_load_##SFX(ptr), npyv_load_##SFX(ptr+npyv_nlanes_##SFX)        \
        );                                                                   \
    }                                                                        \
    NPY_FINLINE void npyv_store_##SFX##x2(                                   \
        npyv_lanetype_##SFX *ptr, npyv_##SFX##x2 v                           \
    ) {                                                                      \
        npyv_##SFX##x2 zip = npyv_zip_##SFX(v.val[0], v.val[1]);             \
        npyv_store_##SFX(ptr, zip.val[0]);                                   \
        npyv_store_##SFX(ptr + npyv_nlanes_##SFX, zip.val[1]);               \
    }

NPYV_IMPL_LASX_MEM_INTERLEAVE(u8, uint8_t);
NPYV_IMPL_LASX_MEM_INTERLEAVE(s8, int8_t)
NPYV_IMPL_LASX_MEM_INTERLEAVE(u16, uint16_t)
NPYV_IMPL_LASX_MEM_INTERLEAVE(s16, int16_t)
NPYV_IMPL_LASX_MEM_INTERLEAVE(u32, uint32_t)
NPYV_IMPL_LASX_MEM_INTERLEAVE(s32, int32_t)
NPYV_IMPL_LASX_MEM_INTERLEAVE(u64, uint64_t)
NPYV_IMPL_LASX_MEM_INTERLEAVE(s64, int64_t)
NPYV_IMPL_LASX_MEM_INTERLEAVE(f32, float)
NPYV_IMPL_LASX_MEM_INTERLEAVE(f64, double)

/*********************************
 * Lookup table
 *********************************/
// uses vector as indexes into a table that contains 32 elements of float32
NPY_FINLINE npyv_f32 npyv_lut32_f32(const float *table, npyv_u32 idx)
{
    const int i0 = __lasx_xvpickve2gr_wu(idx, 0);
    const int i1 = __lasx_xvpickve2gr_wu(idx, 1);
    const int i2 = __lasx_xvpickve2gr_wu(idx, 2);
    const int i3 = __lasx_xvpickve2gr_wu(idx, 3);
    const int i4 = __lasx_xvpickve2gr_wu(idx, 4);
    const int i5 = __lasx_xvpickve2gr_wu(idx, 5);
    const int i6 = __lasx_xvpickve2gr_wu(idx, 6);
    const int i7 = __lasx_xvpickve2gr_wu(idx, 7);
    return npyv_set_f32(table[i0], table[i1], table[i2], table[i3],
                        table[i4], table[i5], table[i6], table[i7]);
}
NPY_FINLINE npyv_u32 npyv_lut32_u32(const npy_uint32 *table, npyv_u32 idx)
{ return npyv_reinterpret_u32_f32(npyv_lut32_f32((const float*)table, idx)); }
NPY_FINLINE npyv_s32 npyv_lut32_s32(const npy_int32 *table, npyv_u32 idx)
{ return npyv_reinterpret_s32_f32(npyv_lut32_f32((const float*)table, idx)); }

// uses vector as indexes into a table that contains 16 elements of float64
NPY_FINLINE npyv_f64 npyv_lut16_f64(const double *table, npyv_u64 idx)
{
    const int i0 = (int)__lasx_xvpickve2gr_du(idx, 0);
    const int i1 = (int)__lasx_xvpickve2gr_du(idx, 1);
    const int i2 = (int)__lasx_xvpickve2gr_du(idx, 2);
    const int i3 = (int)__lasx_xvpickve2gr_du(idx, 3);
    return npyv_set_f64(table[i0], table[i1], table[i2], table[i3]);
}
NPY_FINLINE npyv_u64 npyv_lut16_u64(const npy_uint64 *table, npyv_u64 idx)
{ return npyv_reinterpret_u64_f64(npyv_lut16_f64((const double*)table, idx)); }
NPY_FINLINE npyv_s64 npyv_lut16_s64(const npy_int64 *table, npyv_u64 idx)
{ return npyv_reinterpret_s64_f64(npyv_lut16_f64((const double*)table, idx)); }

#endif // _NPY_SIMD_LASX_MEMORY_H
