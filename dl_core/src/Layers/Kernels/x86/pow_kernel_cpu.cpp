#include "Layers/Kernels/x86/math_kernel_cpu.h"
#include "immintrin.h"
#include <cmath>

namespace cortex_core {

    #define BLOCK_SIZE 8

    void pow_vec_avx256(float* base, float* exp, float* res, const int n) {

    }

    /**
     * Implementation of pow using stl without simd acceleration.
     * @param base the base of the power
     * @param exp the exponent of the power
     * @param res the result tensor
     * @param n number of element in the tensor
     */
    void pow_plain_impl(float* base, float* exp, float* res, const int n) {
        for (int i = 0; i < n; i++) {
            res[i] = std::pow(base[i], exp[i]);
        }
    }

    void pow_scalar_plain_impl(float* base, float exp, float* res, const int n) {
        for (int i = 0; i < n; i++) {
            res[i] = std::pow(base[i], exp);
        }
    }

    void pow_kernel_cpu(const Tensor& a, const Tensor& b, const Tensor& result) {
        if (b.size() == 1) {
            pow_scalar_plain_impl(a.ptr<f32_t>(), *(b.ptr<f32_t>()), result.ptr<f32_t>(), a.size());
        }
        else if (a.size() == 1) {
            pow_scalar_plain_impl(b.ptr<f32_t>(), *(a.ptr<f32_t>()), result.ptr<f32_t>(), b.size());
        }
        else if (a.size() == b.size()) {
            pow_plain_impl(a.ptr<f32_t>(), b.ptr<f32_t>(), result.ptr<f32_t>(), a.size());
        }
        else {
            throw std::runtime_error("pow_kernel_cpu: size mismatch");
        }
    }
}
