#ifndef MATH_KERNEL_CPU_H
#define MATH_KERNEL_CPU_H

#include "Tensor/Tensor.h"

namespace dl_core {
    /**
    * Performs element-wise addition operation between two input tensor and came out with a result tensor
    * @param a the input tensor a
    * @param b the input tensor b
    * @param result the result tensor of a + b
    */
    void add_kernel_cpu(const Tensor& a, const Tensor& b, const Tensor& result);

    /**
    * Performs element-wise subtraction operation between two input tensor and came out with a result tensor
    * @param a the input tensor a
    * @param b the input tensor b
    * @param result the result tensor of a + b
    */
    void sub_kernel_cpu(const Tensor& a, const Tensor& b, const Tensor& result);

    /**
    * Performs element-wise division operation between two input tensor and came out with a result tensor
    * @param a the input tensor a
    * @param b the input tensor b
    * @param result the result tensor of a / b
    * @example let a = [1, 2, 3], b = [2, 4, 6], then c = b / a = [2, 2, 2]
    */
    void div_kernel_cpu(const Tensor& a, const Tensor& b, const Tensor& result);

    /**
    * Performs element-wise multiplication operation between two input tensor and came out with a result tensor
    * @param a the input tensor a
    * @param b the input tensor b
    * @param result the result tensor of a * b
    * @example let a = [1, 2, 3], b = [2, 3, 4], then c = a * b = [2, 6, 12]
    */
    void mul_kernel_cpu(const Tensor& a, const Tensor& b, const Tensor& result);

    /**
    * Performs element-wise power operation with base tensor a and exp tensor b, and came out with a result tensor
    * @param a the input tensor a (base)
    * @param b the input tensor b (exponent)
    * @param result the result tensor of pow(a, b)
    */
    void pow_kernel_cpu(const Tensor& a, const Tensor& b, const Tensor& result);

    /**
    * Performs element-wise exponential operation with input tensor a and came out with a result tensor
    * @param a the input tensor a (base)
    * @param result the result tensor of exp(a)
    */
    void exp_kernel_cpu(const Tensor& a, const Tensor& result);

    /**
    * Performs element-wise logarithmic operation of base e with input tensor a and came out with a result tensor
    * @param a the input tensor a (value)
    * @param result the result tensor of log(a)
    */
    void log_kernel_cpu(const Tensor& a, const Tensor& result);

    /**
    * Performs element-wise logarithmic operation of base 2 with input tensor a and came out with a result tensor
    * @param a the input tensor a (value)
    * @param result the result tensor of log2(a)
    */
    void log2_kernel_cpu(const Tensor& a, const Tensor& result);

    /**
    * Performs element-wise logarithmic operation of base 10 with input tensor a and came out with a result tensor
    * @param a the input tensor a (value)
    * @param result the result tensor of log2(a)
    */
    void log10_kernel_cpu(const Tensor& a, const Tensor& result);

    /**
    * Performs element-wise sin operation with input tensor a and came out with a result tensor
    * @param a the input tensor a (value)
    * @param result the result tensor of sin(a)
    */
    void sin_kernel_cpu(const Tensor& a, const Tensor& result);

    /**
    * Performs element-wise cos operation with input tensor a and came out with a result tensor
    * @param a the input tensor a (value)
    * @param result the result tensor of cos(a)
    */
    void cos_kernel_cpu(const Tensor& a, const Tensor& result);

    /**
    * Performs element-wise tan operation with input tensor a and came out with a result tensor
    * @param a the input tensor a (value)
    * @param result the result tensor of tan(a)
    */
    void tan_kernel_cpu(const Tensor& a, const Tensor& result);

    /**
    * Performs element-wise tan operation with input tensor a and came out with a result tensor
    * @param a the input tensor a (value)
    * @param result the result tensor of tan(a)
    */
    void atan_kernel_cpu(const Tensor& a, const Tensor& result);

    /**
     *
     * @param a
     * @param result
     * @param dim0
     * @param dim1
     * @param in_place
     */
    void transpose_kernel_cpu(const Tensor& a, const Tensor& result, uint32_t dim0, uint32_t dim1, bool in_place);

    /**
     * 
     * @param a
     * @param result 
     */
    void sum_kernel_cpu(const Tensor& a, const Tensor& result);

    /**
     * Performs the matmul operation on cpu
     * @param a tensor a
     * @param b tensor b
     * @param result the result tensor
     */
    void matmul_kernel_cpu(Tensor& a, Tensor& b, Tensor& result);


}

#endif
