#ifndef AVX2_COMMON_EXT_H
#define AVX2_COMMON_EXT_H

#include "immintrin.h"

namespace cortex {
    inline __m256 sum_and_broadcast(__m256 vec) {
        __m256 temp1 = _mm256_hadd_ps(vec, vec);
        __m128 low = _mm256_castps256_ps128(temp1);
        __m128 high = _mm256_extractf128_ps(temp1, 1);
        __m128 sum128 = _mm_add_ps(low, high);
        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        __m256 sum = _mm256_set1_ps(_mm_cvtss_f32(sum128));

        return sum;
    }

    inline float sum_m256(__m256 vec) {
        float res = 0;
        const __m128 low = _mm256_castps256_ps128(vec);
        const auto high = _mm256_extractf128_ps(vec, 1);
        __m128 sum128 = _mm_add_ps(low, high);

        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        res += _mm_cvtss_f32(sum128);

        return res;
    }
}

#endif //AVX2_COMMON_EXT_H
