#include "Layers/Kernels/x86/math_kernel_cpu.h"
#include "cblas.h"

namespace cortex {

#define BLOCK_SIZE 8

    /**
     *
     * @param A
     * @param B
     * @param C
     * @param row_a
     * @param col_a
     * @param col_b
     * @param mat_cnt
     * @param b_flags flags of broadcast: 1 => a need broadcast; 0 => b need broadcast
     */
    void openblas_matmul(const float *A, const float *B, float *C, const int& row_a, const int& col_a, const int& col_b, const int& mat_cnt, const int& b_flags) {
        int matA_size = row_a * col_a;
        int matB_size = col_a * col_b;
        const int matC_size = row_a * col_b;

        if (b_flags == 0)
            matB_size = 0;
        else if (b_flags == 1)
            matA_size = 0;

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

        // Check whether tensor need broadcast
        int flags = -1;
        if (a.size() > b.size())
            flags = 0;
        else if (b.size() > a.size())
            flags = 1;

        openblas_matmul(a.ptr<f32_t>(), b.ptr<f32_t>(), result.ptr<f32_t>(), row_a, col_a, col_b, mat_cnt, flags);
    }

}