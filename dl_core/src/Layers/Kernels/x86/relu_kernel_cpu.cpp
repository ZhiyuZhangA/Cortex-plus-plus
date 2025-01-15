#include "Layers/Kernels/x86/nn_kernel_cpu.h"
#include "immintrin.h"

namespace cortex {

#define BLOCK_SIZE 8

    void relu_avx256(const float* input, float* output, const int& size) {
        const __m256 zeros = _mm256_setzero_ps();
        int i = 0;
        for (; i < size; i += BLOCK_SIZE) {
            __m256 x = _mm256_load_ps(input + i);
            _mm256_store_ps(output + i, _mm256_max_ps(zeros, x));
        }

        // Dealing remaining element
        for (; i < size; i++) {
            output[i] = input[i] > 0 ? input[i] : 0;
        }
    }

    void leaky_relu_avx256(const float* input, float* output, const int& size, const float& ng_slope) {
        __m256 slope_vec = _mm256_set1_ps(ng_slope);
        __m256 zeros = _mm256_setzero_ps();

        int i = 0;
        for (; i < size; i += BLOCK_SIZE) {
            __m256 x = _mm256_load_ps(input + i);
            __m256 mask = _mm256_cmp_ps(x, zeros, _CMP_GE_OS);
            __m256 result = _mm256_blendv_ps(mask, _mm256_mul_ps(x, slope_vec), zeros);

            _mm256_store_ps(output + i, result);
        }

        for (; i < size; i++) {
            output[i] = input[i] > 0 ? input[i] : ng_slope * input[i];
        }
    }

    void relu_kernel_cpu(const Tensor& input, const Tensor& output) {
        relu_avx256(input.ptr<f32_t>(), output.ptr<f32_t>(), input.size());
    }

    /**
     * Computing the derivative of relu function using avx instruction sets
     * @param input the input float array
     * @param output the output float array
     * @param size the size of the float array
     */
    void drelu_avx256(const float* input, float* output, const int& size) {
        int i = 0;
        for (; i < size; i += BLOCK_SIZE) {
            __m256 x = _mm256_load_ps(input + i);
            __m256 mask = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_GT_OQ);
            _mm256_store_ps(output + i, mask);
        }

        // Dealing remaining element
        for (; i < size; i++) {
            output[i] = input[i] > 0 ? 1.0 : 0;
        }
    }

    Tensor drelu_kernel_cpu(const Tensor &input) {
        Tensor ret(input.shape(), input.get_dtype(), input.get_device(), true);
        drelu_avx256(input.ptr<f32_t>(), ret.ptr<f32_t>(), input.size());
        return ret;
    }

    void dleaky_relu_avx256(const float* input, float* output, const int& size, const float& slope) {
        __m256 zeros = _mm256_setzero_ps();
        __m256 ones = _mm256_set1_ps(1);
        __m256 slope_vec = _mm256_set1_ps(slope);

        int i = 0;
        for (; i < size; i += BLOCK_SIZE) {
            __m256 x = _mm256_loadu_ps(&input[i]);
            __m256 mask = _mm256_cmp_ps(x, zeros, _CMP_GT_OQ);
            __m256 positive_part = _mm256_and_ps(mask, ones);
            __m256 negative_part = _mm256_andnot_ps(mask, slope_vec);
            __m256 result = _mm256_or_ps(positive_part, negative_part);

            _mm256_storeu_ps(&output[i], result);
        }

        // Dealing remaining element
        for (; i < size; i++) {
            output[i] = input[i] > 0 ? 1.0f : slope;
        }
    }

    Tensor dleaky_relu_kernel_cpu(const Tensor& input, const float& ng_slope) {
        Tensor ret(input.shape(), input.get_dtype(), input.get_device(), true);
        dleaky_relu_avx256(input.ptr<f32_t>(), ret.ptr<f32_t>(), input.size(), ng_slope);
        return ret;
    }

    void leaky_relu_kernel_cpu(const Tensor& input, const Tensor& output, const float& ng_slope) {
        leaky_relu_avx256(input.ptr<f32_t>(), output.ptr<f32_t>(), input.size(), ng_slope);
    }

}
