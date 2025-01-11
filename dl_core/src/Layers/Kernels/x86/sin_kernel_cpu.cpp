#include "Layers/Kernels/x86/math_kernel_cpu.h"
#include "immintrin.h"
#include <cmath>

namespace dl_core {

#define BLOCK_SIZE 8

    void sin_vec_avx256(float* base, float* exp, float* res, const int n) {

    }

    /**
     * Implementation of sin of base e using stl without simd acceleration.
     * @param value pointer to the input array. Each element represents a value for which the sin will be computed.
     * @param res the result tensor
     * @param n number of element in the tensor
     */
    void sin_plain_impl(const float* value, float* res, const int n) {
        for (int i = 0; i < n; i++) {
            res[i] = std::sin(value[i]);
        }
    }

    void sin_kernel_cpu(const Tensor& a, const Tensor& result) {
        sin_plain_impl(a.ptr<f32_t>(), result.ptr<f32_t>(), a.size());
    }
}
