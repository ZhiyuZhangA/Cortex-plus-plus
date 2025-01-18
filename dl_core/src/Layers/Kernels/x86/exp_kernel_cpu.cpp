#include "Layers/Kernels/x86/math_kernel_cpu.h"
#include "immintrin.h"
#include <cmath>

#include "avx2_extension/avx2_math_ext.h"

namespace cortex {

#define BLOCK_SIZE 8

    void exp_vec_avx256(const float* value, float* res, const int n) {
        int i = 0;
        const int len = (int)(n / BLOCK_SIZE) * BLOCK_SIZE;
        for (; i + 7 < len; i+=BLOCK_SIZE) {
            __m256 _a = _mm256_load_ps(value + i);
            _mm256_store_ps(res + i, _mm256_exp_ps(_a));
        }

        // Remaining element
        for (i = n - n % BLOCK_SIZE; i < n; i++) {
            res[i] = std::exp(value[i]);
        }
    }

    /**
     * Implementation of exp using stl without simd acceleration.
     * @param value pointer to the input array. Each element represents a value for which the exponential will be computed.
     * @param res the result tensor
     * @param n number of element in the tensor
     */
    void exp_plain_impl(const float* value, float* res, const int n) {
        for (int i = 0; i < n; i++) {
            res[i] = std::exp(value[i]);
        }
    }

    void exp_kernel_cpu(const Tensor& a, const Tensor& result) {
        exp_vec_avx256(a.ptr<f32_t>(), result.ptr<f32_t>(), a.size());
    }
}
