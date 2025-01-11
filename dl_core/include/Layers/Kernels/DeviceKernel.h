#ifndef DEVICE_KERNEL_H
#define DEVICE_KERNEL_H
#include "Tensor/Tensor.h"

/**
 * The core logics for binary operator add, sub, mul and div is that the kernel will optimize for element-wise operation
 * and operation with scalar specifically, while for two operands without same shape, the broadcast logics will be addressed
 * in the upper layer (tensor function).
 */
namespace cortex_core {
    typedef void (*add_kernel)(const Tensor& a, const Tensor& b, const Tensor& result);
    typedef void (*sub_kernel)(const Tensor& a, const Tensor& b, const Tensor& result);
    typedef void (*mul_kernel)(const Tensor& a, const Tensor& b, const Tensor& result);
    typedef void (*div_kernel)(const Tensor& a, const Tensor& b, const Tensor& result);
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

    typedef void (*transpose_kernel)(const Tensor& a, const Tensor& result, uint32_t dim0, uint32_t dim1, bool in_place);

    add_kernel get_add_kernel(DeviceType deviceType);
    sub_kernel get_sub_kernel(DeviceType deviceType);
    mul_kernel get_mul_kernel(DeviceType deviceType);
    div_kernel get_div_kernel(DeviceType deviceType);
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

}


#endif
