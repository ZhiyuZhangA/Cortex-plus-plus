#ifndef AVX2_MATH_EXT_H
#define AVX2_MATH_EXT_H

#include <immintrin.h>

#define SUPPORT_FMA_

namespace cortex {

    /**
     * 
     * @param x
     * @return 
     */
    inline __m256 _mm256_exp_ps(__m256 x) {
        const __m256 p0 = _mm256_set1_ps(1.0);
        const __m256 p1 = _mm256_set1_ps(0.5000000000f);
        const __m256 p2 = _mm256_set1_ps(0.166666666666666667f);
        const __m256 p3 = _mm256_set1_ps(0.0416666667f);
        const __m256 p4 = _mm256_set1_ps(0.0083333333f);
        const __m256 p5 = _mm256_set1_ps(0.0013888889f);
        const __m256 p6 = _mm256_set1_ps(1.9841269841e-4);
        const __m256 p7 = _mm256_set1_ps(2.4801587302e-5);
        // const __m256 p8 = _mm256_set1_ps(2.7557319224e-6);

        __m256 y = _mm256_add_ps(p0, x);
        __m256 term = x;

#ifdef SUPPORT_FMA_
        term = _mm256_mul_ps(term, x);
        y = _mm256_fmadd_ps(term, p1, y);
        term = _mm256_mul_ps(term, x);
        y = _mm256_fmadd_ps(term, p2, y);
        term = _mm256_mul_ps(term, x);
        y = _mm256_fmadd_ps(term, p3, y);
        term = _mm256_mul_ps(term, x);
        y = _mm256_fmadd_ps(term, p4, y);
        term = _mm256_mul_ps(term, x);
        y = _mm256_fmadd_ps(term, p5, y);
        term = _mm256_mul_ps(term, x);
        y = _mm256_fmadd_ps(term, p6, y);
        term = _mm256_mul_ps(term, x);
        y = _mm256_fmadd_ps(term, p7, y);

#else
        term = _mm256_mul_ps(term, x);
        y = _mm256_add_ps(y, _mm256_mul_ps(term, p1));
        term = _mm256_mul_ps(term, x);
        y = _mm256_add_ps(y, _mm256_mul_ps(term, p2));
        term = _mm256_mul_ps(term, x);
        y = _mm256_add_ps(y, _mm256_mul_ps(term, p3));
        term = _mm256_mul_ps(term, x);
        y = _mm256_add_ps(y, _mm256_mul_ps(term, p4));
        term = _mm256_mul_ps(term, x);
        y = _mm256_add_ps(y, _mm256_mul_ps(term, p5));
        term = _mm256_mul_ps(term, x);
        y = _mm256_add_ps(y, _mm256_mul_ps(term, p6));
        term = _mm256_mul_ps(term, x);
        y = _mm256_add_ps(y, _mm256_mul_ps(term, p7));
#endif

        return y;
    }

    const float ps256_1[8] __attribute__((aligned(32))) = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    const float ps256_min_norm_pos[8] __attribute__((aligned(32))) = {1.17549435e-38f, 1.17549435e-38f, 1.17549435e-38f, 1.17549435e-38f, 1.17549435e-38f, 1.17549435e-38f, 1.17549435e-38f, 1.17549435e-38f};
    const float ps256_inv_mant_mask[8] __attribute__((aligned(32))) = {2139095039.0f, 2139095039.0f, 2139095039.0f, 2139095039.0f, 2139095039.0f, 2139095039.0f, 2139095039.0f, 2139095039.0f};
    const float ps256_0p5[8] __attribute__((aligned(32))) = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    const float ps256_cephes_SQRTHF[8] __attribute__((aligned(32))) = {0.70710678118f, 0.70710678118f, 0.70710678118f, 0.70710678118f, 0.70710678118f, 0.70710678118f, 0.70710678118f, 0.70710678118f};
    const float ps256_cephes_log_p0[8] __attribute__((aligned(32))) = {7.0376836292E-2f, 7.0376836292E-2f, 7.0376836292E-2f, 7.0376836292E-2f, 7.0376836292E-2f, 7.0376836292E-2f, 7.0376836292E-2f, 7.0376836292E-2f};
    const float ps256_cephes_log_p1[8] __attribute__((aligned(32))) = {-1.1514610310E-1f, -1.1514610310E-1f, -1.1514610310E-1f, -1.1514610310E-1f, -1.1514610310E-1f, -1.1514610310E-1f, -1.1514610310E-1f, -1.1514610310E-1f};
    const float ps256_cephes_log_p2[8] __attribute__((aligned(32))) = {1.1676998740E-1f, 1.1676998740E-1f, 1.1676998740E-1f, 1.1676998740E-1f, 1.1676998740E-1f, 1.1676998740E-1f, 1.1676998740E-1f, 1.1676998740E-1f};
    const float ps256_cephes_log_p3[8] __attribute__((aligned(32))) = {-1.2420140846E-1f, -1.2420140846E-1f, -1.2420140846E-1f, -1.2420140846E-1f, -1.2420140846E-1f, -1.2420140846E-1f, -1.2420140846E-1f, -1.2420140846E-1f};
    const float ps256_cephes_log_p4[8] __attribute__((aligned(32))) = {1.4249322787E-1f, 1.4249322787E-1f, 1.4249322787E-1f, 1.4249322787E-1f, 1.4249322787E-1f, 1.4249322787E-1f, 1.4249322787E-1f, 1.4249322787E-1f};
    const float ps256_cephes_log_p5[8] __attribute__((aligned(32))) = {-1.6668057665E-1f, -1.6668057665E-1f, -1.6668057665E-1f, -1.6668057665E-1f, -1.6668057665E-1f, -1.6668057665E-1f, -1.6668057665E-1f, -1.6668057665E-1f};
    const float ps256_cephes_log_p6[8] __attribute__((aligned(32))) = {2.0000714765E-1f, 2.0000714765E-1f, 2.0000714765E-1f, 2.0000714765E-1f, 2.0000714765E-1f, 2.0000714765E-1f, 2.0000714765E-1f, 2.0000714765E-1f};
    const float ps256_cephes_log_p7[8] __attribute__((aligned(32))) = {-2.4999993993E-1f, -2.4999993993E-1f, -2.4999993993E-1f, -2.4999993993E-1f, -2.4999993993E-1f, -2.4999993993E-1f, -2.4999993993E-1f, -2.4999993993E-1f};
    const float ps256_cephes_log_p8[8] __attribute__((aligned(32))) = {3.3333331174E-1f, 3.3333331174E-1f, 3.3333331174E-1f, 3.3333331174E-1f, 3.3333331174E-1f, 3.3333331174E-1f, 3.3333331174E-1f, 3.3333331174E-1f};
    const float ps256_cephes_log_q1[8] __attribute__((aligned(32))) = {-2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f, -2.12194440e-4f};
    const float ps256_cephes_log_q2[8] __attribute__((aligned(32))) = {0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f, 0.693359375f};
    const int pi32_256_0x7f[8] __attribute__((aligned(32))) = {0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f, 0x7f};

    #define LOAD_PS256(ptr) _mm256_loadu_ps(ptr)
    #define LOAD_PI256(ptr) _mm256_loadu_si256((const __m256i*)(ptr))

    inline __m256 _mm256_log_ps(__m256 x) {
        printf("Input: %f %f %f %f %f %f %f %f\n", x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);

        __m256i imm0;
        const __m256 one = LOAD_PS256(ps256_1);

        // Mask for invalid inputs (x <= 0)
        const __m256 invalid_mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);
        printf("Invalid mask: %f %f %f %f %f %f %f %f\n", invalid_mask[0], invalid_mask[1], invalid_mask[2], invalid_mask[3], invalid_mask[4], invalid_mask[5], invalid_mask[6], invalid_mask[7]);

        // Ensure input x is in normalized range
        x = _mm256_max_ps(x, LOAD_PS256(ps256_min_norm_pos));
        printf("Normalized input: %f %f %f %f %f %f %f %f\n", x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);

        // Extract exponent and normalize mantissa
        imm0 = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
        x = _mm256_and_ps(x, LOAD_PS256(ps256_inv_mant_mask));
        printf("Mantissa normalized: %f %f %f %f %f %f %f %f\n", x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
        x = _mm256_or_ps(x, LOAD_PS256(ps256_0p5));

        // Adjust exponent
        imm0 = _mm256_sub_epi32(imm0, LOAD_PI256(pi32_256_0x7f));
        __m256 e = _mm256_cvtepi32_ps(imm0);
        e = _mm256_add_ps(e, one);
        printf("Exponent adjusted: %f %f %f %f %f %f %f %f\n", e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7]);

        // Mask for values less than SQRTHF
        const __m256 mask = _mm256_cmp_ps(x, LOAD_PS256(ps256_cephes_SQRTHF), _CMP_LT_OS);
        const __m256 tmp = _mm256_and_ps(x, mask);
        x = _mm256_sub_ps(x, one);
        e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
        x = _mm256_add_ps(x, tmp);
        printf("Adjusted values: %f %f %f %f %f %f %f %f\n", x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);

        // Polynomial approximation for log(1 + x)
        const __m256 z = _mm256_mul_ps(x, x);
        __m256 y = _mm256_fmadd_ps(LOAD_PS256(ps256_cephes_log_p0), x, LOAD_PS256(ps256_cephes_log_p1));
        y = _mm256_fmadd_ps(y, x, LOAD_PS256(ps256_cephes_log_p2));
        y = _mm256_fmadd_ps(y, x, LOAD_PS256(ps256_cephes_log_p3));
        y = _mm256_fmadd_ps(y, x, LOAD_PS256(ps256_cephes_log_p4));
        y = _mm256_fmadd_ps(y, x, LOAD_PS256(ps256_cephes_log_p5));
        y = _mm256_fmadd_ps(y, x, LOAD_PS256(ps256_cephes_log_p6));
        y = _mm256_fmadd_ps(y, x, LOAD_PS256(ps256_cephes_log_p7));
        y = _mm256_fmadd_ps(y, x, LOAD_PS256(ps256_cephes_log_p8));
        y = _mm256_mul_ps(y, x);
        y = _mm256_mul_ps(y, z);
        printf("Polynomial result: %f %f %f %f %f %f %f %f\n", y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);

        // Combine polynomial result with exponent contribution
        y = _mm256_fmadd_ps(e, LOAD_PS256(ps256_cephes_log_q1), y);
        y = _mm256_fnmadd_ps(z, LOAD_PS256(ps256_0p5), y);

        // Final result with adjustments
        const __m256 final_tmp = _mm256_fmadd_ps(e, LOAD_PS256(ps256_cephes_log_q2), y);
        x = _mm256_add_ps(x, final_tmp);
        x = _mm256_or_ps(x, invalid_mask);  // Set NaN for invalid inputs
        printf("Final result: %f %f %f %f %f %f %f %f\n", x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);

        return x;
    }



    
}

#endif //AVX2_MATH_EXT_H
