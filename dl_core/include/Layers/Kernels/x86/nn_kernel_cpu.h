#ifndef NN_KERNEL_CPU_H
#define NN_KERNEL_CPU_H
#include "Tensor/Tensor.h"

namespace cortex {
    void linear_kernel_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& output);
}

#endif
