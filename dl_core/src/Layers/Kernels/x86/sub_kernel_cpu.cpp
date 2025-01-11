#include "Layers/Kernels/x86/math_kernel_cpu.h"
#include "immintrin.h"

namespace cortex_core {

#define BLOCK_SIZE 8

    void sub_vec_avx256(float* a, float* b, float* c, const int n) {
        const int len = (int)(n / BLOCK_SIZE) * BLOCK_SIZE;
        for (int i = 0; i < len; i+=BLOCK_SIZE) {
            __m256 _a = _mm256_load_ps(a);
            __m256 _b = _mm256_load_ps(b);

            _mm256_store_ps(c + i, _mm256_sub_ps(_a, _b));
        }

        // Remaining element
        for (int i = n - n % BLOCK_SIZE; i < n; i++) {
            c[i] = a[i] - b[i];
        }
    }

    void sub_scalar_avx256(float* a, const float* b, float* c, const int n) {
        __m256 _b = _mm256_set1_ps(b[0]);
        const int len = (int)(n / BLOCK_SIZE) * BLOCK_SIZE;
        for (int i = 0; i < len; i+=BLOCK_SIZE) {
            __m256 _a = _mm256_load_ps(a);

            _mm256_store_ps(c + i, _mm256_sub_ps(_a, _b));
        }

        // Remaining element
        for (int i = n - n % BLOCK_SIZE; i < n; i++) {
            c[i] = a[i] - b[0];
        }
    }

    void sub_kernel_cpu(const Tensor& a, const Tensor& b, const Tensor& result) {
        if (b.size() == 1) {
            sub_scalar_avx256(a.ptr<f32_t>(), b.ptr<f32_t>(), result.ptr<f32_t>(), a.size());
        }
        else if (a.size() == 1) {
            sub_scalar_avx256(b.ptr<f32_t>(), a.ptr<f32_t>(), result.ptr<f32_t>(), b.size());
        }
        else if (a.size() == b.size()) {
            sub_vec_avx256(a.ptr<f32_t>(), b.ptr<f32_t>(), result.ptr<f32_t>(), a.size());
        }
        else {
            throw std::runtime_error("sub_kernel_cpu: size mismatch");
        }
    }
}
