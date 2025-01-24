#include <iostream>
#include "Tensor/Tensor.h"
#include "immintrin.h"
#include "avx2_extension/avx2_math_ext.h"
#include "avx2_extension/avx2_common_ext.h"

namespace cortex {

#define BLOCK_SIZE 8

    void cross_entropy_avx256(const float* label, const float* prediction, float* output, const int& n) {

        // Create temporary memory for output
        const auto temp_output = static_cast<float*>(malloc(n * sizeof(float)));

        // Calculate the softmax first
        __m256 sum_vec = _mm256_setzero_ps();
        float max_val = *std::max_element(prediction, prediction + n);
        __m256 max = _mm256_set1_ps(max_val);
        const int len = (n / BLOCK_SIZE) * BLOCK_SIZE;
        const int remaining_idx = n - n % BLOCK_SIZE;
        for (int i = 0; i < len; i+=BLOCK_SIZE) {
            const __m256 x = _mm256_load_ps(prediction + i);
            const __m256 exp = _mm256_exp_ps(_mm256_sub_ps(x, max));
            sum_vec = _mm256_add_ps(sum_vec, exp);
            _mm256_store_ps(temp_output + i, exp);
        }

        float sum_scalar = sum_m256(sum_vec);
        // Dealing with the remaining elements
        for (int i = remaining_idx; i < n; i++) {
            temp_output[i] = std::exp(prediction[i] - max_val);
            sum_scalar += temp_output[i];
        }

        const __m256 sum = _mm256_set1_ps(sum_scalar);

        // Divide all elements by the sum of exps
        for (int i = 0; i < len; i+=BLOCK_SIZE) {
            const __m256 x = _mm256_load_ps(temp_output + i);
            _mm256_store_ps(temp_output + i, _mm256_div_ps(x, sum));
        }

        // Dealing with the remaining elements
        for (int i = remaining_idx; i < n; i++) {
            temp_output[i] /= sum_scalar;
        }

        for (int i = 0; i < n; i++) {
            temp_output[i] = std::log(temp_output[i]);
        }

        // Multiply with label
        for (int i = 0; i < len; i+=BLOCK_SIZE) {
            __m256 _a = _mm256_load_ps(label + i);
            __m256 _b = _mm256_load_ps(temp_output + i);

            _mm256_store_ps(temp_output + i, _mm256_mul_ps(_a, _b));
        }

        // Remaining element
        for (int i = n - n % BLOCK_SIZE; i < n; i++) {
            temp_output[i] = temp_output[i] * label[i];
        }

        // Sum
        auto ce_sum = _mm256_setzero_ps();
        for (int i = 0; i < len; i+=BLOCK_SIZE) {
            ce_sum = _mm256_add_ps(ce_sum, _mm256_load_ps(temp_output + i));
        }

        free(temp_output);
        output[0] = -sum_m256(ce_sum);
    }

    void cross_entropy_loss_kernel_cpu(const Tensor& label, const Tensor& prediction, const Tensor& output) {
        cross_entropy_avx256(label.ptr<f32_t>(), prediction.ptr<f32_t>(), output.ptr<f32_t>(), label.size());
    }

}

