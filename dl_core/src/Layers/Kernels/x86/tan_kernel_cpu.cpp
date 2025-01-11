#include "Layers/Kernels/x86/math_kernel_cpu.h"
#include "immintrin.h"
#include <cmath>

namespace cortex_core {

#define BLOCK_SIZE 8

    void tan_vec_avx256(float* base, float* exp, float* res, const int n) {

    }

    /**
     * Implementation of tan of base e using stl without simd acceleration.
     * @param value pointer to the input array. Each element represents a value for which the tan will be computed.
     * @param res the result tensor
     * @param n number of element in the tensor
     */
    void tan_plain_impl(const float* value, float* res, const int n) {
        for (int i = 0; i < n; i++) {
            res[i] = std::cos(value[i]);
        }
    }

    void tan_kernel_cpu(const Tensor& a, const Tensor& result) {
        tan_plain_impl(a.ptr<f32_t>(), result.ptr<f32_t>(), a.size());
    }
}
