#include <iostream>
#include "Tensor/Tensor.h"
#include "immintrin.h"

namespace cortex {

#define BLOCK_SIZE 8

    void mse_none_reduction_avx256(const float* label, const float* prediction, float* output, const int& n) {
        const int len = (n / BLOCK_SIZE) * BLOCK_SIZE;
        for (int i = 0; i < len; i+=BLOCK_SIZE) {
            __m256 _a = _mm256_load_ps(label + i);
            __m256 _b = _mm256_load_ps(prediction + i);
            __m256 _c = _mm256_sub_ps(_a, _b);

            _mm256_store_ps(output + i, _mm256_mul_ps(_c, _c));
        }

        // Remaining Elements
        for (int i = n - n % BLOCK_SIZE; i < n; i++) {
            output[i] = (label[i] - prediction[i]) * (label[i] - prediction[i]);
        }
    }

    void mse_sum_avx256(const float* label, const float* prediction, float* output, const int& n) {
        float total_sum = 0;
        __m256 vec_sum = _mm256_setzero_ps();

        const int len = (n / BLOCK_SIZE) * BLOCK_SIZE;
        for (int i = 0; i < len; i+=BLOCK_SIZE) {
            __m256 _a = _mm256_load_ps(label + i);
            __m256 _b = _mm256_load_ps(prediction + i);
            __m256 _c = _mm256_sub_ps(_a, _b);

            vec_sum = _mm256_add_ps(vec_sum, _mm256_mul_ps(_c, _c));
        }

        // Summing the m256
        const __m128 low = _mm256_castps256_ps128(vec_sum);
        const __m128 high = _mm256_extractf128_ps(vec_sum, 1);
        __m128 sum128 = _mm_add_ps(low, high);

        sum128 = _mm_hadd_ps(sum128, sum128);
        sum128 = _mm_hadd_ps(sum128, sum128);
        total_sum += _mm_cvtss_f32(sum128);

        // Remaining Elements
        for (int i = n - n % BLOCK_SIZE; i < n; i++) {
            total_sum += (label[i] - prediction[i]) * (label[i] - prediction[i]);
        }

        output[0] = total_sum;
    }

    void mse_mean_avx256(const float* label, const float* prediction, float* output, const int& n) {
        mse_sum_avx256(label, prediction, output, n);
        output[0] /= n;
    }

    void mse_loss_kernel_cpu(const Tensor& label, const Tensor& prediction, const Tensor& output, uint8_t mode) {
        if (mode == 0) {
            mse_none_reduction_avx256(label.ptr<f32_t>(), prediction.ptr<f32_t>(), output.ptr<f32_t>(), prediction.size());
        }
        else if (mode == 1) {
            mse_sum_avx256(label.ptr<f32_t>(), prediction.ptr<f32_t>(), output.ptr<f32_t>(), prediction.size());
        }
        else if (mode == 2) {
            mse_mean_avx256(label.ptr<f32_t>(), prediction.ptr<f32_t>(), output.ptr<f32_t>(), prediction.size());
        }
        else {
            throw std::invalid_argument("MSELoss Mode Error: Unknown mode input!");
        }
    }

}