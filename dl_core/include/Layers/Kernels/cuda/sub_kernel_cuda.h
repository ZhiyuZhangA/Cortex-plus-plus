#ifndef SUB_KERNEL_CUDA_H
#define SUB_KERNEL_CUDA_H

#include "Tensor/Tensor.h"

namespace cortex_core {
    void sub_kernel_cuda(const Tensor& a, const Tensor& b, const Tensor& result);
}



#endif
