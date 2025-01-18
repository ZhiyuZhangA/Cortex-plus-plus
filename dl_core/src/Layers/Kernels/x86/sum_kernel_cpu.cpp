#include "Layers/Kernels/x86/math_kernel_cpu.h"
#include "immintrin.h"
#include <cmath>

namespace cortex {

#define BLOCK_SIZE 8

    void sum_avx256(const float* data, const int& n, float* result) {
        __m256 sum_vec = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 vec = _mm256_loadu_ps(&data[i]);
            sum_vec = _mm256_add_ps(sum_vec, vec);
        }

        __m128 low = _mm256_castps256_ps128(sum_vec);
        __m128 high = _mm256_extractf128_ps(sum_vec, 1);
        __m128 sum128 = _mm_add_ps(low, high);

        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        float sum = _mm_cvtss_f32(sum128);

        for (i = n - n % BLOCK_SIZE; i < n; ++i) {
            sum += data[i];
        }

        *result = sum;
    }

    void sum_kernel_cpu(const Tensor& a, const Tensor& result) {
        sum_avx256(a.ptr<f32_t>(), a.size(), result.ptr<f32_t>());
    }
}
