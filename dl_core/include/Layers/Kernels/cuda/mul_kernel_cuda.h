//
// Created by zzy on 2025/1/3.
//

#ifndef MUL_KERNEL_CUDA_H
#define MUL_KERNEL_CUDA_H

#include "Tensor/Tensor.h"

namespace dl_core {
    void mul_kernel_cuda(const Tensor& a, const Tensor& b, const Tensor& result);
}


#endif //MUL_KERNEL_CUDA_H
