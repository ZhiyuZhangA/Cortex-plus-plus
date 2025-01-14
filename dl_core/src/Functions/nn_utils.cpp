#include "Functions/nn_utils.h"

#include <iostream>

#include "DLEngine/DLEngine.h"
#include "Layers/Kernels/DeviceKernel.h"
#include "Layers/nn/LinearLayer.h"

namespace cortex {
    Tensor FLinear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
        // Check whether the shape match or not
        if (input.shape()[input.dim() - 1] !=  weight.shape()[weight.dim() - 1]) {
            throw std::invalid_argument("Input and Weight must have the same size");
        }

        // Set the output shape
        std::vector<uint32_t> output_shape = input.shape();
        output_shape[output_shape.size() - 1] = weight.shape()[weight.dim() - 2];

        Tensor ret(output_shape, input.get_dtype(), input.get_device(), input.enable_grad());

        // Directly compute the linear part
        get_linear_kernel(input.get_device())(input, weight, bias, ret);

        if (DLEngine::is_grad_mode()) {
            ret.grad_func() = std::make_shared<LinearLayer>(input.get_dtype(), input.get_device(), false);
            ret.grad_func()->add_input(input);
            ret.grad_func()->add_input(weight);
            ret.grad_func()->add_input(bias);
            ret.grad_func()->add_output(ret);
        }

        return ret;
    }

    Tensor FLinear(const Tensor& input, const Tensor& weight) {
        // Check whether the shape match or not
        if (input.shape()[input.dim() - 1] !=  weight.shape()[weight.dim() - 1]) {
            throw std::invalid_argument("Input and Weight must have the same size");
        }

        // Set the output shape
        std::vector<uint32_t> output_shape = input.shape();
        output_shape[output_shape.size() - 1] = weight.shape()[weight.dim() - 2];
        Tensor ret(output_shape, input.get_dtype(), input.get_device(), input.enable_grad());

        // Directly compute the linear part
        get_linear_no_bias_kernel(input.get_device())(input, weight, ret);

        if (DLEngine::is_grad_mode()) {
            ret.grad_func() = std::make_shared<LinearLayer>(input.get_dtype(), input.get_device(), false);
            ret.grad_func()->add_input(input);
            ret.grad_func()->add_input(weight);
            ret.grad_func()->add_output(ret);
        }

        return ret;
    }

}
