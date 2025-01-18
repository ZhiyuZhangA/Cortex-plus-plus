#include "Layers/Kernels/x86/nn_kernel_cpu.h"
#include "immintrin.h"
#include <cmath>

namespace cortex {

#define BLOCK_SIZE 8

    void sigmoid_avx256(const float* input, float* output, const int& n) {
        int i = 0;
        const __m256 one = _mm256_set1_ps(1.0f);
        for (; i + 7 < n; i+=BLOCK_SIZE) {
            const __m256 x_vec = _mm256_load_ps(input + i);
            const __m256 reciprocal = _mm256_div_ps(one, x_vec);
            const __m256 res = _mm256_div_ps(one, _mm256_add_ps(one, reciprocal));
            _mm256_store_ps(output + i, res);
        }

        // Remaining Elements
        for (i = n - n % BLOCK_SIZE; i < n; i++) {
            output[i] = 1.0 / (1.0 + 1.0 / std::exp(input[i]));
        }
    }

    void sigmoid_kernel_cpu(const Tensor& input, const Tensor& result) {
        sigmoid_avx256(input.ptr<f32_t>(), result.ptr<f32_t>(), input.size());
    }
}
