#include "Layers/Kernels/x86/math_kernel_cpu.h"
#include "immintrin.h"
#include "avx2_extension/avx2_common_ext.h"

namespace cortex {

#define BLOCK_SIZE 8

    void sum_avx256(const float* data, const int& n, float* result) {
        __m256 sum_vec = _mm256_setzero_ps();
        int i = 0;
        for (; i + 7 < n; i += 8) {
            __m256 vec = _mm256_loadu_ps(&data[i]);
            sum_vec = _mm256_add_ps(sum_vec, vec);
        }

        float sum = sum_m256(sum_vec);

        for (i = n - n % BLOCK_SIZE; i < n; ++i) {
            sum += data[i];
        }

        *result = sum;
    }

    void sum_kernel_cpu(const Tensor& a, const Tensor& result) {
        sum_avx256(a.ptr<f32_t>(), a.size(), result.ptr<f32_t>());
    }
}
