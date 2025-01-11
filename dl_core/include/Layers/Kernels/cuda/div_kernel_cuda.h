//
// Created by zzy on 2025/1/3.
//

#ifndef DIV_KERNEL_CUDA_H
#define DIV_KERNEL_CUDA_H

#include "Tensor/Tensor.h"

namespace cortex {
    void div_kernel_cuda(const Tensor& a, const Tensor& b, const Tensor& result);
}


#endif //DIV_KERNEL_CUDA_H
