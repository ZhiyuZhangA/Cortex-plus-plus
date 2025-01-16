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
}

#endif //AVX2_MATH_EXT_H
