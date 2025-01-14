#include "Layers/Kernels/x86/nn_kernel_cpu.h"
#include "immintrin.h"

namespace cortex {

#define BLOCK_SIZE 8

    void relu_avx256(const float* input, float* output, const int& size) {
        const __m256 zero = _mm256_setzero_ps();
        int i = 0;
        for (i = 0; i < size; i += BLOCK_SIZE) {
            __m256 x = _mm256_load_ps(input + i);
            _mm256_store_ps(output + i, _mm256_max_ps(zero, x));
        }

        // Dealing remaining element
        for (; i < size; i++) {
            output[i] = output[i] > 0 ? output[i] : 0;
        }
    }

    void relu_kernel_cpu(const Tensor& input, const Tensor& output) {
        relu_avx256(input.ptr<f32_t>(), output.ptr<f32_t>(), input.size());
    }

}
