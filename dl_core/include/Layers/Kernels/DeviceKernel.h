#ifndef DEVICE_KERNEL_H
#define DEVICE_KERNEL_H
#include "Tensor/Tensor.h"

/**
 * The core logics for binary operator add, sub, mul and div is that the kernel will optimize for element-wise operation
 * and operation with scalar specifically, while for two operands without same shape, the broadcast logics will be addressed
 * in the upper layer (tensor function).
 */
namespace cortex {
    typedef void (*add_kernel)(const Tensor& a, const Tensor& b, const Tensor& result);
    typedef void (*sub_kernel)(const Tensor& a, const Tensor& b, const Tensor& result);
    typedef void (*mul_kernel)(const Tensor& a, const Tensor& b, const Tensor& result);
    typedef void (*div_kernel)(const Tensor& a, const Tensor& b, const Tensor& result);
    typedef void (*matmul_kernel)(const Tensor& a, const Tensor& b, const Tensor& result);
    typedef void (*pow_kernel)(const Tensor& a, const Tensor& b, const Tensor& result);
    typedef void (*exp_kernel)(const Tensor& a, const Tensor& result);
    typedef void (*log_kernel)(const Tensor& a, const Tensor& result);
    typedef void (*log2_kernel)(const Tensor& a, const Tensor& result);
    typedef void (*log10_kernel)(const Tensor& a, const Tensor& result);
    typedef void (*sin_kernel)(const Tensor& a, const Tensor& result);
    typedef void (*cos_kernel)(const Tensor& a, const Tensor& result);
    typedef void (*tan_kernel)(const Tensor& a, const Tensor& result);
    typedef void (*atan_kernel)(const Tensor& a, const Tensor& result);
    typedef void (*asin_kernel)(const Tensor& a, const Tensor& result);
    typedef void (*acos_kernel)(const Tensor& a, const Tensor& result);
    typedef void (*sum_kernel)(const Tensor& a, const Tensor& result);

    typedef void (*linear_kernel)(const Tensor& input, const Tensor& weight, const Tensor& bias, const Tensor& output);
    typedef void (*linear_kernel_no_bias)(const Tensor& input, const Tensor& weight, const Tensor& output);
    typedef void (*relu_kernel)(const Tensor& input, const Tensor& output);
    typedef Tensor (*drelu_kernel)(const Tensor& input);
    typedef void (*leaky_relu_kernel)(const Tensor& input, const Tensor& output, const float& ng_slope);
    typedef Tensor (*dleaky_relu_kernel)(const Tensor& input, const float& ng_slope);
    typedef void (*sigmoid_kernel)(const Tensor& input, const Tensor& result);
    typedef void (*mse_loss_kernel)(const Tensor& label, const Tensor& prediction, const Tensor& output, uint8_t mode);

    typedef void (*transpose_kernel)(const Tensor& a, const Tensor& result, uint32_t dim0, uint32_t dim1, bool in_place);


    add_kernel get_add_kernel(DeviceType deviceType);
    sub_kernel get_sub_kernel(DeviceType deviceType);
    mul_kernel get_mul_kernel(DeviceType deviceType);
    div_kernel get_div_kernel(DeviceType deviceType);
    matmul_kernel get_matmul_kernel(DeviceType deviceType);
    pow_kernel get_pow_kernel(DeviceType deviceType);
    exp_kernel get_exp_kernel(DeviceType deviceType);
    log_kernel get_log_kernel(DeviceType deviceType);
    log2_kernel get_log2_kernel(DeviceType deviceType);
    log10_kernel get_log10_kernel(DeviceType deviceType);
    sin_kernel get_sin_kernel(DeviceType deviceType);
    cos_kernel get_cos_kernel(DeviceType deviceType);
    tan_kernel get_tan_kernel(DeviceType deviceType);
    atan_kernel get_atan_kernel(DeviceType deviceType);

    transpose_kernel get_transpose_kernel(DeviceType deviceType);
    sum_kernel get_sum_kernel(DeviceType deviceType);

    linear_kernel get_linear_kernel(DeviceType deviceType);
    linear_kernel_no_bias get_linear_no_bias_kernel(DeviceType deviceType);
    relu_kernel get_relu_kernel(DeviceType deviceType);
    drelu_kernel get_drelu_kernel(DeviceType deviceType);
    leaky_relu_kernel get_leaky_relu_kernel(DeviceType deviceType);
    dleaky_relu_kernel get_dleaky_relu_kernel(DeviceType deviceType);
    sigmoid_kernel get_sigmoid_kernel(DeviceType deviceType);

    mse_loss_kernel get_mse_loss_kernel(DeviceType deviceType);

}


#endif
