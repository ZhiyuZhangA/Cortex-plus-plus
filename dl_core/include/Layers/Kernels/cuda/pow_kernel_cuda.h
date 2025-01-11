#ifndef POW_KERNEL_CUDA_H
#define POW_KERNEL_CUDA_H

#include "Tensor/Tensor.h"

namespace cortex {
    /**
    * Performs element-wise power operation with base tensor a and exp tensor b, and came out with a result tensor on cuda
    * @param a the input tensor a (base)
    * @param b the input tensor b (exponent)
    * @param result the result tensor of pow(a, b)
    */
    void pow_kernel_cuda(const Tensor& a, const Tensor& b, const Tensor& result);

}



#endif //POW_KERNEL_CUDA_H
