/*
 * Payne-Hanek range reduction for f64 sin/cos.
 *
 * The fast path in npyv_sincos.h uses Cody-Waite reduction with a three-part
 * π/2 split, which is precise only when y = round(x · 2/π) is exactly
 * representable as an int — roughly |x| < 2^20 with our particular constants.
 * Beyond that the original kernel falls back to scalar libm.
 *
 * This header replaces the libm fallback with our own Payne-Hanek
 * implementation. The reduction is scalar per-lane (a precomputed bit table
 * of 2/π is gathered based on x's exponent) but the polynomial that runs on
 * the reduced argument afterward stays vectorized — so we still get the
 * f64 SIMD speedup on the polynomial side for the rare large-|x| case.
 *
 * Algorithm (FDLIBM e_rem_pio2.c / k_rem_pio2.c, simplified for nx=1,
 * prec=2):
 *
 *   1. Split |x| into a 24-bit-mantissa chunk x_chunk and an exponent e0.
 *   2. Index the precomputed 2/π table at the chunk corresponding to bit
 *      position e0; load enough chunks to cover both the integer part of
 *      x·2/π (which determines the quadrant) and ~53 bits of the
 *      fractional part (which becomes the reduced argument).
 *   3. Multiply the x mantissa by each 2/π chunk to get partial products
 *      q[0..jk]; each multiplication is exact in f64 (24-bit × 24-bit
 *      fits in 53 bits).
 *   4. Distill q[] into 24-bit integer pieces iq[] from the high end
 *      down; pull out the integer part of x·2/π for the quadrant.
 *   5. Look at the "half" bit (bit -1 of the fractional part). If set,
 *      the rounded integer was actually one higher than what the integer
 *      part suggested, so flip the sign convention and complement iq[].
 *   6. Multiply the surviving fraction back by π/2 (also stored as
 *      24-bit-aligned chunks) to recover the reduced argument in
 *      double-double form. The "hi" component becomes our reduced r.
 *
 * Reference: Payne, M. & Hanek, R., "Radian Reduction for Trigonometric
 * Functions," SIGNUM Newsletter, 1983. Algorithm structure and table data
 * follow Sun's FDLIBM, public-domain.
 */
#ifndef _NPY_UMATH_PAYNE_HANEK_F64_H
#define _NPY_UMATH_PAYNE_HANEK_F64_H

#include <math.h>
#include <stdint.h>

/* 2/π as 24-bit integer chunks. Each entry is bits at offset i·24 below
 * the binary point. From FDLIBM ipio2[] (public domain).
 *
 * 66 chunks × 24 bits = 1584 bits of 2/π. Enough to handle |x| up to
 * 2^1024 (f64 max) plus a healthy margin (~500 bits).
 */
static const int32_t npyv__payne_hanek_two_over_pi[] = {
    0xA2F983, 0x6E4E44, 0x1529FC, 0x2757D1, 0xF534DD, 0xC0DB62,
    0x95993C, 0x439041, 0xFE5163, 0xABDEBB, 0xC561B7, 0x246E3A,
    0x424DD2, 0xE00649, 0x2EEA09, 0xD1921C, 0xFE1DEB, 0x1CB129,
    0xA73EE8, 0x8235F5, 0x2EBB44, 0x84E99C, 0x7026B4, 0x5F7E41,
    0x3991D6, 0x398353, 0x39F49C, 0x845F8B, 0xBDF928, 0x3B1FF8,
    0x97FFDE, 0x05980F, 0xEF2F11, 0x8B5A0A, 0x6D1F6D, 0x367ECF,
    0x27CB09, 0xB74F46, 0x3F669E, 0x5FEA2D, 0x7527BA, 0xC7EBE5,
    0xF17B3D, 0x0739F7, 0x8A5292, 0xEA6BFB, 0x5FB11F, 0x8D5D08,
    0x560330, 0x46FC7B, 0x6BABF0, 0xCFBC20, 0x9AF436, 0x1DA9E3,
    0x91615E, 0xE61B08, 0x659985, 0x5F14A0, 0x68408D, 0xFFD880,
    0x4D7327, 0x310606, 0x1556CA, 0x73A8C9, 0x60E27B, 0xC08C6B,
};

/* π/2 split into successive 24-bit-aligned chunks. From FDLIBM PIo2[]. */
static const double npyv__payne_hanek_pi_over_2[] = {
    1.57079625129699707031e+00, /* 0x3FF921FB40000000 */
    7.54978941586159635335e-08, /* 0x3E74442D00000000 */
    5.39030252995776476554e-15, /* 0x3CF8469880000000 */
    3.28200341580835373941e-22, /* 0x3B78CC5160000000 */
    1.27065575308067607349e-29, /* 0x39F01B8380000000 */
    1.22933308981111328932e-36, /* 0x387A252080000000 */
    2.73370053816464559624e-44, /* 0x36E3822280000000 */
    2.16741683877804819444e-51, /* 0x3569F31D00000000 */
};

#define NPYV__PH_TWO24    16777216.0
#define NPYV__PH_TWON24   5.9604644775390625e-08

/*
 * Reduce x to r ∈ [-π/4, π/4] with quadrant n in {0,1,2,3}. Assumes
 * x is finite and |x| ≥ 2^-50 (caller masks tiny inputs); returns the
 * quadrant and writes r to *r_out.
 */
static inline int
npyv__payne_hanek_reduce(double x, double *r_out)
{
    /* Sign handling: do reduction on |x|, propagate sign onto the
     * reduced argument and quadrant at the end. */
    int sign = signbit(x);
    double ax = sign ? -x : x;

    /* Decompose |x| into 24-bit chunks tx[0..2] and an exponent offset e0
     * such that ax = (tx[0] + tx[1]·2^-24 + tx[2]·2^-48) · 2^e0. tx[0] is
     * the most significant chunk (in [1, 2^24)). e0 is the binary exponent
     * of tx[0].
     *
     * Following FDLIBM e_rem_pio2.c: scale ax so the exponent field is
     * exactly 1046 (giving an unbiased exponent of 23 → value in [2^23,
     * 2^24)), then peel off integer chunks. */
    union { double f; uint64_t u; } u;
    u.f = ax;
    int32_t hx = (int32_t)(u.u >> 32);
    int32_t ix = hx & 0x7FFFFFFF;
    int32_t e0 = (ix >> 20) - 1046;   /* ilogb(ax) - 23 */
    /* Replace ax's exponent with 1046 (biased), keeping mantissa. */
    u.u = (u.u & 0xFFFFFFFFULL) |
          ((uint64_t)(uint32_t)(ix - (e0 << 20))) << 32;
    double z = u.f;
    double tx[3];
    tx[0] = (double)(int32_t)z;
    z = (z - tx[0]) * NPYV__PH_TWO24;
    tx[1] = (double)(int32_t)z;
    z = (z - tx[1]) * NPYV__PH_TWO24;
    tx[2] = z;
    int nx = 3;
    while (nx > 1 && tx[nx - 1] == 0.0) nx--;
    /* FDLIBM's __kernel_rem_pio2 takes e0 as the exponent of tx[0] itself
     * (the highest chunk). That matches our e0 above. */

    /* ---- begin __kernel_rem_pio2 ---- */
    /* jk = 4 chunks of result (enough for f64) */
    const int jk = 4;
    const int jp = 4;
    int jx = nx - 1;
    int jv = (e0 - 3) / 24;
    if (jv < 0) jv = 0;
    int q0 = e0 - 24 * (jv + 1);

    /* Load 2/π chunks into f[0..jx+jk] */
    double f[20];
    int i, j;
    j = jv - jx;
    int m = jx + jk;
    for (i = 0; i <= m; i++, j++) {
        f[i] = (j < 0) ? 0.0
                       : (double)npyv__payne_hanek_two_over_pi[j];
    }

    /* Partial products q[0..jk] = tx · f shifted */
    double q[20];
    for (i = 0; i <= jk; i++) {
        double fw = 0.0;
        for (j = 0; j <= jx; j++) {
            fw += tx[j] * f[jx + i - j];
        }
        q[i] = fw;
    }

    int jz = jk;
    int32_t iq[20];
    int n = 0;
    int ih = 0;
    int carry;
    int k;

recompute:
    /* Distill q[] into 24-bit integer pieces iq[] */
    z = q[jz];
    for (i = 0, j = jz; j > 0; i++, j--) {
        double fw = (double)((int32_t)(NPYV__PH_TWON24 * z));
        iq[i] = (int32_t)(z - NPYV__PH_TWO24 * fw);
        z = q[j - 1] + fw;
    }

    /* z = (q[0] + carry-ins) at exponent q0 — pull off integer part for n */
    z = ldexp(z, q0);
    z -= 8.0 * floor(z * 0.125);    /* keep z mod 8 (so n mod 8 is fine) */
    n = (int32_t)z;
    z -= (double)n;
    ih = 0;
    if (q0 > 0) {
        i = (iq[jz - 1] >> (24 - q0));
        n += i;
        iq[jz - 1] -= i << (24 - q0);
        ih = iq[jz - 1] >> (23 - q0);
    } else if (q0 == 0) {
        ih = iq[jz - 1] >> 23;
    } else if (z >= 0.5) {
        ih = 2;
    }

    if (ih > 0) {
        /* Reduced fraction > 0.5 → bump quadrant and complement iq */
        n += 1;
        carry = 0;
        for (i = 0; i < jz; i++) {
            j = iq[i];
            if (carry == 0) {
                if (j != 0) {
                    carry = 1;
                    iq[i] = 0x1000000 - j;
                }
            } else {
                iq[i] = 0xFFFFFF - j;
            }
        }
        if (q0 > 0) {
            switch (q0) {
            case 1: iq[jz - 1] &= 0x7FFFFF; break;
            case 2: iq[jz - 1] &= 0x3FFFFF; break;
            }
        }
        if (ih == 2) {
            z = 1.0 - z;
            if (carry != 0) z -= ldexp(1.0, q0);
        }
    }

    /* If the fraction is exactly zero, may need more terms */
    if (z == 0.0) {
        j = 0;
        for (i = jz - 1; i >= jk; i--) j |= iq[i];
        if (j == 0) {
            for (k = 1; iq[jk - k] == 0; k++) { /* count trailing zeros */ }
            for (i = jz + 1; i <= jz + k; i++) {
                f[jx + i] = (double)npyv__payne_hanek_two_over_pi[jv + i];
                double fw = 0.0;
                for (j = 0; j <= jx; j++) fw += tx[j] * f[jx + i - j];
                q[i] = fw;
            }
            jz += k;
            goto recompute;
        }
    }

    /* Chop trailing zero pieces; renormalize z back into iq[] if it's large */
    if (z == 0.0) {
        jz -= 1; q0 -= 24;
        while (iq[jz] == 0) { jz--; q0 -= 24; }
    } else {
        z = ldexp(z, -q0);
        if (z >= NPYV__PH_TWO24) {
            double fw = (double)((int32_t)(NPYV__PH_TWON24 * z));
            iq[jz] = (int32_t)(z - NPYV__PH_TWO24 * fw);
            jz += 1;
            q0 += 24;
            iq[jz] = (int32_t)fw;
        } else {
            iq[jz] = (int32_t)z;
        }
    }

    /* Convert iq[] back to floating-point pieces */
    double fw = ldexp(1.0, q0);
    for (i = jz; i >= 0; i--) {
        q[i] = fw * (double)iq[i];
        fw *= NPYV__PH_TWON24;
    }

    /* Multiply by π/2 (chunked) and accumulate */
    double fq[20];
    for (i = jz; i >= 0; i--) {
        fw = 0.0;
        for (k = 0; k <= jp && k <= jz - i; k++) {
            fw += npyv__payne_hanek_pi_over_2[k] * q[i + k];
        }
        fq[jz - i] = fw;
    }

    /* Sum fq[] for the reduced argument; prec=2 here so we just want
     * the high part. (FDLIBM also computes a lo part for ULP precision;
     * for our polynomial it's not needed since the sin/cos polynomial
     * is already designed around a single-double reduced argument.) */
    fw = 0.0;
    for (i = jz; i >= 0; i--) fw += fq[i];

    /* ih == 0 → reduced fraction was in [0, 0.5], result is +fw.
     * Otherwise we complemented, so result is -fw. */
    double r = (ih == 0) ? fw : -fw;

    /* Final sign / quadrant. If the original x was negative, both flip:
     *   sin(-x) = -sin(x)  → negate r, negate quadrant's sign meaning
     *   cos(-x) =  cos(x)
     * The quadrant logic in the polynomial caller handles this by
     * looking at bit 1 of n (the negate bit) and bit 0 (the cos-vs-sin
     * bit). For sin, negating x flips bit 1 of n by adding 2 mod 4
     * after we negate r. Simplest: negate r and add 2 to n if sign. */
    if (sign) {
        r = -r;
        n = -n;
    }

    *r_out = r;
    return n & 3;
}

#endif /* _NPY_UMATH_PAYNE_HANEK_F64_H */
