//
// Created by zzy on 2024/12/28.
//

#ifndef ADD_KERNEL_CUDA_H
#define ADD_KERNEL_CUDA_H

#include "Tensor/Tensor.h"

namespace cortex_core {
    void add_kernel_cuda(const Tensor& a, const Tensor& b, const Tensor& result);
}



#endif //ADD_KERNEL_CUDA_H
