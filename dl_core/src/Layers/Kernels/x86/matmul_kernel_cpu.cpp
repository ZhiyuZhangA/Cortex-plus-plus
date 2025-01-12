#include "Layers/Kernels/x86/math_kernel_cpu.h"
#include "cblas.h"

namespace cortex {

#define BLOCK_SIZE 8

    void openblas_matmul(const float *A, const float *B, float *C, const int& row_a, const int& col_a, const int& col_b, const int& mat_cnt) {
        const int matA_size = row_a * col_a;
        const int matB_size = col_a * col_b;
        const int matC_size = row_a * col_b;
        for (int i = 0; i < mat_cnt; i++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row_a, col_b, col_a, 1.0f, A + i * matA_size, col_a, B + i * matB_size, col_b, 0.0f, C + i * matC_size, col_b);
        }
    }

    void matmul_kernel_cpu(const Tensor& a, const Tensor& b, const Tensor& result) {
        // Check the size
        const int row_a = a.shape()[a.shape().size() - 2];
        const int col_a = a.shape().back();
        const int col_b = b.shape().back();
        const int mat_cnt = a.size() / (row_a * col_a);
        openblas_matmul(a.ptr<f32_t>(), b.ptr<f32_t>(), result.ptr<f32_t>(), row_a, col_a, col_b, mat_cnt);
    }

}