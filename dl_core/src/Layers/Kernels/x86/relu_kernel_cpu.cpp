#include "Layers/Kernels/x86/nn_kernel_cpu.h"
#include "immintrin.h"

namespace cortex {

#define BLOCK_SIZE 8

    void relu_avx256(const float* input, float* output, const int& size) {
        const __m256 zero = _mm256_setzero_ps();
        int i = 0;
        for (; i < size; i += BLOCK_SIZE) {
            __m256 x = _mm256_load_ps(input + i);
            _mm256_store_ps(output + i, _mm256_max_ps(zero, x));
        }

        // Dealing remaining element
        for (; i < size; i++) {
            output[i] = input[i] > 0 ? input[i] : 0;
        }
    }

    void leaky_relu_avx256(const float* input, float* output, const int& size, const float& ng_slope) {
        __m256 slope_vec = _mm256_set1_ps(ng_slope);
        __m256 zero = _mm256_setzero_ps();

        int i = 0;
        for (; i < size; i += BLOCK_SIZE) {
            __m256 x = _mm256_load_ps(input + i);
            __m256 mask = _mm256_cmp_ps(x, zero, _CMP_GE_OS);
            __m256 result = _mm256_blendv_ps(mask, _mm256_mul_ps(x, slope_vec), zero);

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

    void leaky_relu_kernel_cpu(const Tensor& input, const Tensor& output, const float& ng_slope) {
        leaky_relu_avx256(input.ptr<f32_t>(), output.ptr<f32_t>(), input.size(), ng_slope);
    }

}
