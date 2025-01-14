#include "Layers/Kernels/x86/nn_kernel_cpu.h"
#include "cblas.h"
#include "immintrin.h"


namespace cortex {

#define BLOCK_SIZE 8

    /**
     * Compute the linear layer XW^T + b with bias term
     * @param input (*, in_features) => (ri,ci)
     * @param weight (out_features, in_features) => (rw,ci)
     * @param bias the bias term of linear operation (out_features)
     * @param output the output tensor of linear operation (*, out_features), (ri,rw)
     * @param ri the row of input tensor
     * @param ci the column of input tensor or the row of the weight tensor
     * @param rw the row of weight tensor or the column of output tensor
     * @param mat_cnt the matrix count
     */
    void linear_bias_blas(const float* input, const float* weight, float* bias, float* output, const int& ri, const int& ci, const int& rw, const int& mat_cnt) {
        // first compute xw^T
        const int mat_x_size = ri * ci;
        const int mat_y_size = ri * rw;

        for (int i = 0; i < mat_cnt; i++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ri, rw, ci, 1.0f, input + i * mat_x_size, ci, weight, ci, 0.0f, output + i * mat_y_size, rw);

            const int mat_offset = mat_y_size * i;
            // For all rows
            for (int j = 0; j < ri; j++) {
                // For all columns
                if (rw >= BLOCK_SIZE) {
                    for (int k = 0; k < rw; k+=BLOCK_SIZE) {
                        __m256 _r = _mm256_load_ps(output + mat_offset + j * rw + k);
                        __m256 _b = _mm256_load_ps(bias + k);
                        _mm256_store_ps(output + mat_offset + j * rw + k, _mm256_add_ps(_b, _r));
                    }
                }

                // Remaining element
                for (int k = rw - rw % BLOCK_SIZE; k < rw; k++) {
                    output[mat_offset + j * rw + k] += bias[k];
                }
            }
        }
    }

    /**
     * Computes the linear layer XW^T without bias term
     * @param input (*, in_features) => (ri,ci)
     * @param weight (out_features, in_features) => (rw,ci)
     * @param output the output tensor of linear operation (*, out_features), (ri,rw)
     * @param ri the row of input tensor
     * @param ci the column of input tensor or the row of the weight tensor
     * @param rw the row of weight tensor or the column of output tensor
     * @param mat_cnt the matrix count
     */
    void linear_no_bias_blas(const float* input, const float* weight, float* output, const int& ri, const int& ci, const int& rw, const int& mat_cnt) {
        // first compute xw^T
        const int mat_x_size = ri * ci;
        const int mat_y_size = ri * rw;

        for (int i = 0; i < mat_cnt; i++) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ri, rw, ci, 1.0f, input + i * mat_x_size, ci, weight, ci, 0.0f, output + i * mat_y_size, rw);
        }
    }

    void linear_kernel_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& output) {
        const int ri = input.shape()[input.dim() - 2];
        const int ci = input.shape()[input.dim() - 1];
        const int rw = weight.shape()[weight.dim() - 2];
        const int mat_cnt = input.size() / (ri * ci);
        linear_bias_blas(input.ptr<f32_t>(), weight.ptr<f32_t>(), bias.ptr<f32_t>(), output.ptr<f32_t>(), ri, ci, rw, mat_cnt);
    }

    void linear_kernel_no_bias_cpu(const Tensor& input, const Tensor& weight, const Tensor& output) {
        const int ri = input.shape()[input.dim() - 2];
        const int ci = input.shape()[input.dim() - 1];
        const int rw = weight.shape()[weight.dim() - 2];
        const int mat_cnt = input.size() / (ri * ci);

        linear_no_bias_blas(input.ptr<f32_t>(), weight.ptr<f32_t>(), output.ptr<f32_t>(), ri, ci, rw, mat_cnt);
    }


}