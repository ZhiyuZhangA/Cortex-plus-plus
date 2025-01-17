#ifndef NN_KERNEL_CPU_H
#define NN_KERNEL_CPU_H
#include "Tensor/Tensor.h"

namespace cortex {
    void linear_kernel_cpu(const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& output);
    void linear_kernel_no_bias_cpu(const Tensor& input, const Tensor& weight, const Tensor& output);
    void relu_kernel_cpu(const Tensor& input, const Tensor& output);
    Tensor drelu_kernel_cpu(const Tensor& input);
    void leaky_relu_kernel_cpu(const Tensor& input, const Tensor& output, const float& ng_slope);
    Tensor dleaky_relu_kernel_cpu(const Tensor& input, const float& ng_slope);

    void mse_loss_kernel_cpu(const Tensor& label, const Tensor& prediction, const Tensor& output, uint8_t mode);
}

#endif
