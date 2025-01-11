#include "Layers/Kernels/x86/math_kernel_cpu.h"
#include "immintrin.h"
#include <cmath>

namespace cortex {

#define BLOCK_SIZE 8

    void cos_vec_avx256(float* base, float* exp, float* res, const int n) {

    }

    /**
     * Implementation of cos of base e using stl without simd acceleration.
     * @param value pointer to the input array. Each element represents a value for which the cos will be computed.
     * @param res the result tensor
     * @param n number of element in the tensor
     */
    void cos_plain_impl(const float* value, float* res, const int n) {
        for (int i = 0; i < n; i++) {
            res[i] = std::cos(value[i]);
        }
    }

    void cos_kernel_cpu(const Tensor& a, const Tensor& result) {
        cos_plain_impl(a.ptr<f32_t>(), result.ptr<f32_t>(), a.size());
    }
}
