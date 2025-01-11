#ifndef TRANPOSE_KERNEL_CUDA_H
#define TRANPOSE_KERNEL_CUDA_H


#include "Tensor/Tensor.h"
namespace cortex {
    void transpose_kernel_cuda(const Tensor& a, const Tensor& result, uint32_t dim0, uint32_t dim1, bool in_place);
}


#endif //TRANPOSE_KERNEL_CUDA_H
