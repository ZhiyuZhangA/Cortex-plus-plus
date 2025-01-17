#include "Layers/Kernels/x86/math_kernel_cpu.h"
#include "immintrin.h"
#include <cmath>

namespace cortex {

#define BLOCK_SIZE 8

    /**
     * Implementation of log of base e using stl without simd acceleration.
     * @param value pointer to the input array. Each element represents a value for which the logarithm will be computed.
     * @param res the result tensor
     * @param n number of element in the tensor
     */
    void log_plain_impl(const float* value, float* res, const int n) {
        for (int i = 0; i < n; i++) {
            res[i] = std::log(value[i]);
        }
    }

    /**
     * Implementation of log of base 2 using stl without simd acceleration.
     * @param value pointer to the input array. Each element represents a value for which the logarithm will be computed.
     * @param res the result tensor
     * @param n number of element in the tensor
     */
    void log2_plain_impl(const float* value, float* res, const int n) {
        for (int i = 0; i < n; i++) {
            res[i] = std::log2(value[i]);
        }
    }

    /**
     * Implementation of log of base 10 using stl without simd acceleration.
     * @param value pointer to the input array. Each element represents a value for which the logarithm will be computed.
     * @param res the result tensor
     * @param n number of element in the tensor
     */
    void log10_plain_impl(const float* value, float* res, const int n) {
        for (int i = 0; i < n; i++) {
            res[i] = std::log10(value[i]);
        }
    }

    void log_kernel_cpu(const Tensor& a, const Tensor& result) {
        log_plain_impl(a.ptr<f32_t>(), result.ptr<f32_t>(), a.size());
        // log_vec_avx256(a.ptr<f32_t>(), result.ptr<f32_t>(), a.size());
    }

    void log2_kernel_cpu(const Tensor& a, const Tensor& result) {
        log2_plain_impl(a.ptr<f32_t>(), result.ptr<f32_t>(), a.size());
    }

    void log10_kernel_cpu(const Tensor& a, const Tensor& result) {
        log10_plain_impl(a.ptr<f32_t>(), result.ptr<f32_t>(), a.size());
    }
}
