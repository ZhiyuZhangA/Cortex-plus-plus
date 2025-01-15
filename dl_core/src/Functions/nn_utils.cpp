#include "Functions/nn_utils.h"
#include "DLEngine/DLEngine.h"
#include "Layers/Kernels/DeviceKernel.h"
#include "Layers/nn/LinearLayer.h"
#include "Layers/nn/ReLuLayer.h"

namespace cortex {
    Tensor FLinear(const Tensor& input, const Tensor& weight, const Tensor& bias) {
        // Check whether the shape match or not
        if (input.shape()[input.dim() - 1] !=  weight.shape()[weight.dim() - 1]) {
            throw std::invalid_argument("FLinear Error: Input tensor and Weight tensor must be compatible for matrix multiplication!");
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
            throw std::invalid_argument("FLinear Error: Input tensor and Weight tensor must be compatible for matrix multiplication!");
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

    Tensor FReLu(const Tensor& input) {
        Tensor ret(input.shape(), input.get_dtype(), input.get_device(), true);
        get_relu_kernel(input.get_device())(input, ret);

        if (DLEngine::is_grad_mode()) {
            ret.grad_func() = std::make_shared<ReLuLayer>(input.get_dtype(), input.get_device(), false);
            ret.grad_func()->add_input(input);
            ret.grad_func()->add_output(ret);
        }

        return ret;
    }

    Tensor FLeakyReLu(const Tensor &input, const float& ng_slopes) {
        Tensor ret(input.shape(), input.get_dtype(), input.get_device(), true);
        get_leaky_relu_kernel(input.get_device())(input, ret, ng_slopes);

        if (DLEngine::is_grad_mode()) {
            ret.grad_func() = std::make_shared<LeakyReLuLayer>(input.get_dtype(), input.get_device(), false);
            ret.grad_func()->add_input(input);
            ret.grad_func()->add_param(ng_slopes);
            ret.grad_func()->add_output(ret);
        }

        return ret;
    }
}
