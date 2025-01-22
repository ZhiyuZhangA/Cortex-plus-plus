#include "Layers/Kernels/x86/nn_kernel_cpu.h"
#include "avx2_extension/avx2_math_ext.h"
#include "avx2_extension/avx2_common_ext.h"
#include "immintrin.h"
#include <cmath>

namespace cortex {

#define BLOCK_SIZE 8

    /**
     *
     * @param input
     * @param output
     * @param n
     */
    void softmax_avx256(const float* input, float* output, const int& n) {
        // Apply exponential function to the input and store the result into the output using simd
        __m256 sum_vec = _mm256_setzero_ps();
        const int len = (n / BLOCK_SIZE) * BLOCK_SIZE;
        const int remaining_idx = n - n % BLOCK_SIZE;
        for (int i = 0; i < len; i+=BLOCK_SIZE) {
            const __m256 x = _mm256_load_ps(input + i);
            const __m256 exp = _mm256_exp_ps(x);
            sum_vec = _mm256_add_ps(sum_vec, exp);
            _mm256_store_ps(output + i, exp);
        }

        float sum_scalar = sum_m256(sum_vec);
        // Dealing with the remaining elements
        for (int i = remaining_idx; i < n; i++) {
            output[i] = std::exp(input[i]);
            sum_scalar += output[i];
        }

        const __m256 sum = _mm256_set1_ps(sum_scalar);

        // Divide all elements by the sum of exps
        for (int i = 0; i < len; i+=BLOCK_SIZE) {
            const __m256 x = _mm256_load_ps(output + i);
            _mm256_store_ps(output + i, _mm256_div_ps(x, sum));
        }

        // Dealing with the remaining elements
        for (int i = remaining_idx; i < n; i++) {
            output[i] /= sum_scalar;
        }
    }

    void softmax_kernel_cpu(const Tensor& input, const Tensor& output) {
        // Only Support input of dimension of 2
        if (input.dim() > 1 && input.dim() <= 2) {
            const int batch = input.shape()[0];
            const int data_cnt = input.shape()[1];
            auto input_ptr = input.ptr<f32_t>();
            auto output_ptr = output.ptr<f32_t>();
            for (int i = 0; i < batch; i++) {
                softmax_avx256(input_ptr, output_ptr, data_cnt);
                input_ptr += data_cnt;
                output_ptr += data_cnt;
            }
        }
        else if (input.dim() == 1) {
            softmax_avx256(input.ptr<f32_t>(), output.ptr<f32_t>(), input.size());
        }
    }
}

