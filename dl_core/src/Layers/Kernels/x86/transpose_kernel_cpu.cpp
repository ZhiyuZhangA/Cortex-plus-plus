#include "Layers/Kernels/x86/math_kernel_cpu.h"
#include <cblas.h>

namespace dl_core {

    void transpose_mkl(const float* input, float* output, const std::vector<uint32_t>& shape, const uint32_t dim0, const uint32_t dim1, const bool in_place) {
        // Determine whether the dim0 and dim1 are coming from the row and column
        if (dim0 + dim1 == 1) {
            const int rows = shape[shape.size() - 2];
            const int cols = shape[shape.size() - 1];

            // Count numbers of matrix of rows and cols
            int mat_cnt = 1;
            for (int i = shape.size() - 2; i >= 0; i--) {
                mat_cnt *= shape[i];
            }

            const int size = rows * cols;

            // modification occurs in-place since openblas doesn't support in-place matrix modification
            if (in_place) {
                std::vector<float> tmp(size);

                for (int i = 0; i < mat_cnt; i++) {
                    cblas_somatcopy(CblasRowMajor, CblasTrans, rows, cols, 1.0f, input + i * size, cols, tmp.data(), rows);
                    std::copy_n(tmp.data(), size, output + i * size);
                }
            }
            else {
                for (int i = 0; i < mat_cnt; i++) {
                    cblas_somatcopy(CblasRowMajor, CblasTrans, rows, cols, 1.0f, input + i * size, cols, output + i * size, rows);
                }
            }
        }
        else {

        }
    }

    void transpose_kernel_cpu(const Tensor& a, const Tensor& result, uint32_t dim0, uint32_t dim1, bool in_place) {
        transpose_mkl(a.ptr<f32_t>(), result.ptr<f32_t>(), a.shape(), dim0, dim1, in_place);
    }

}

