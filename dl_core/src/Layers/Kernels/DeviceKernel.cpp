#include "Layers/Kernels/DeviceKernel.h"
#include "Layers/Kernels/x86/math_kernel_cpu.h"
#include "Layers/Kernels/cuda/add_kernel_cuda.h"
#include "Layers/Kernels/cuda/div_kernel_cuda.h"
#include "Layers/Kernels/cuda/mul_kernel_cuda.h"
#include "Layers/Kernels/cuda/sub_kernel_cuda.h"
#include "Layers/Kernels/cuda/tranpose_kernel_cuda.h"

namespace cortex {
    add_kernel get_add_kernel(const DeviceType deviceType) {
        if (deviceType == DeviceType::cpu)
            return add_kernel_cpu;
        else if (deviceType == DeviceType::cuda)
            return add_kernel_cuda;
        else
            throw std::runtime_error("Unknown device type!");
    }

    sub_kernel get_sub_kernel(const DeviceType deviceType) {
        if (deviceType == DeviceType::cpu)
            return sub_kernel_cpu;
        else if (deviceType == DeviceType::cuda)
            return sub_kernel_cuda;
        else
            throw std::runtime_error("Unknown device type!");
    }

    mul_kernel get_mul_kernel(const DeviceType deviceType) {
        if (deviceType == DeviceType::cpu)
            return mul_kernel_cpu;
        else if (deviceType == DeviceType::cuda)
            return mul_kernel_cuda;
        else
            throw std::runtime_error("Unknown device type!");
    }

    div_kernel get_div_kernel(const DeviceType deviceType) {
        if (deviceType == DeviceType::cpu)
            return div_kernel_cpu;
        else if (deviceType == DeviceType::cuda)
            return div_kernel_cuda;
        else
            throw std::runtime_error("Unknown device type!");
    }

    matmul_kernel get_matmul_kernel(const DeviceType deviceType) {
        if (deviceType == DeviceType::cpu)
            return matmul_kernel_cpu;
        // else if (deviceType == DeviceType::cuda)
        //     return atan_kernel_cuda;
        else
            throw std::runtime_error("Unknown device type!");
    }

    pow_kernel get_pow_kernel(const DeviceType deviceType) {
        if (deviceType == DeviceType::cpu)
            return pow_kernel_cpu;
        // else if (deviceType == DeviceType::cuda)
        //     return exp_kernel_cuda;
        else
            throw std::runtime_error("Unknown device type!");
    }

    exp_kernel get_exp_kernel(const DeviceType deviceType) {
        if (deviceType == DeviceType::cpu)
            return exp_kernel_cpu;
        // else if (deviceType == DeviceType::cuda)
        //     return exp_kernel_cuda;
        else
            throw std::runtime_error("Unknown device type!");
    }

    log_kernel get_log_kernel(const DeviceType deviceType) {
        if (deviceType == DeviceType::cpu)
            return log_kernel_cpu;
        // else if (deviceType == DeviceType::cuda)
        //     return log_kernel_cuda;
        else
            throw std::runtime_error("Unknown device type!");
    }

    log_kernel get_log2_kernel(DeviceType deviceType) {
        if (deviceType == DeviceType::cpu)
            return log2_kernel_cpu;
        // else if (deviceType == DeviceType::cuda)
        //     return log_kernel_cuda;
        else
            throw std::runtime_error("Unknown device type!");
    }

    log10_kernel get_log10_kernel(DeviceType deviceType) {
        if (deviceType == DeviceType::cpu)
            return log10_kernel_cpu;
        // else if (deviceType == DeviceType::cuda)
        //     return log_kernel_cuda;
        else
            throw std::runtime_error("Unknown device type!");
    }

    sin_kernel get_sin_kernel(const DeviceType deviceType) {
        if (deviceType == DeviceType::cpu)
            return sin_kernel_cpu;
        // else if (deviceType == DeviceType::cuda)
        //     return sin_kernel_cuda;
        else
            throw std::runtime_error("Unknown device type!");
    }

    cos_kernel get_cos_kernel(const DeviceType deviceType) {
        if (deviceType == DeviceType::cpu)
            return cos_kernel_cpu;
        // else if (deviceType == DeviceType::cuda)
        //     return cos_kernel_cuda;
        else
            throw std::runtime_error("Unknown device type!");
    }

    tan_kernel get_tan_kernel(const DeviceType deviceType) {
        if (deviceType == DeviceType::cpu)
            return tan_kernel_cpu;
        // else if (deviceType == DeviceType::cuda)
        //     return tan_kernel_cuda;
        else
            throw std::runtime_error("Unknown device type!");
    }

    atan_kernel get_atan_kernel(const DeviceType deviceType) {
        if (deviceType == DeviceType::cpu)
            return atan_kernel_cpu;
        // else if (deviceType == DeviceType::cuda)
        //     return atan_kernel_cuda;
        else
            throw std::runtime_error("Unknown device type!");
    }

    sum_kernel get_sum_kernel(const DeviceType deviceType) {
        if (deviceType == DeviceType::cpu)
            return sum_kernel_cpu;
        // else if (deviceType == DeviceType::cuda)
        //     return atan_kernel_cuda;
        else
            throw std::runtime_error("Unknown device type!");
    }

    transpose_kernel get_transpose_kernel(const DeviceType deviceType) {
        if (deviceType == DeviceType::cpu)
            return transpose_kernel_cpu;
        else if (deviceType == DeviceType::cuda)
            return transpose_kernel_cuda;
        else
            throw std::runtime_error("Unknown device type!");
    }


}
