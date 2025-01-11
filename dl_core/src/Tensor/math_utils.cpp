#include "Tensor/math_utils.h"
#include "DLEngine/DLEngine.h"
#include "Layers/CosLayer.h"
#include "Layers/PowLayer.h"
#include "Layers/ExpLayer.h"
#include "Layers/LogLayer.h"
#include "Layers/SinLayer.h"
#include "Layers/TanLayer.h"
#include "Layers/Kernels/DeviceKernel.h"

namespace cortex_core {
    Tensor pow(const Tensor& base, const Tensor& n) {
        Tensor ret(base.shape(), base.get_dtype(), base.get_device(), base.enable_grad() || n.enable_grad());
        get_pow_kernel(base.get_device())(base, n, ret);

        if (DLEngine::is_grad_mode()) {
            ret.grad_func() = std::make_shared<PowLayer>(base.get_dtype(), base.get_device(), false);
            ret.grad_func()->add_input(base);
            ret.grad_func()->add_input(n);
            ret.grad_func()->add_output(ret);
        }

        return ret;
    }

    Tensor pow(const Tensor& base, const f32_t& n) {
        const Tensor op_s(base.shape(), base.get_dtype(), base.get_device(), false);
        op_s.fill_(n);

        return pow(base, op_s);
    }

    Tensor exp(const Tensor& x) {
        Tensor ret(x.shape(), x.get_dtype(), x.get_device(), x.enable_grad());
        get_exp_kernel(x.get_device())(x, ret);
        if (DLEngine::is_grad_mode()) {
            ret.grad_func() = std::make_shared<ExpLayer>(x.get_dtype(), x.get_device(), false);
            ret.grad_func()->add_input(x);
            ret.grad_func()->add_output(ret);
        }

        return ret;
    }

    Tensor log(const Tensor& x) {
        Tensor ret(x.shape(), x.get_dtype(), x.get_device(), x.enable_grad());
        get_log_kernel(x.get_device())(x, ret);
        if (DLEngine::is_grad_mode()) {
            ret.grad_func() = std::make_shared<LogLayer>(x.get_dtype(), x.get_device(), false);
            ret.grad_func()->add_input(x);
            // Add the Euler's number e for the natural log
            Tensor base({1}, x.get_dtype(), x.get_device(), x.enable_grad());
            base.fill_(std::exp(1));
            ret.grad_func()->add_input(base);
            ret.grad_func()->add_output(ret);
        }

        return ret;
    }

    Tensor log2(const Tensor& x) {
        Tensor ret(x.shape(), x.get_dtype(), x.get_device(), x.enable_grad());
        get_log2_kernel(x.get_device())(x, ret);
        if (DLEngine::is_grad_mode()) {
            ret.grad_func() = std::make_shared<LogLayer>(x.get_dtype(), x.get_device(), false);
            ret.grad_func()->add_input(x);
            // Add the base 2 for the log
            Tensor base({1}, x.get_dtype(), x.get_device(), x.enable_grad());
            base.fill_(2);
            ret.grad_func()->add_input(base);
            ret.grad_func()->add_output(ret);
        }

        return ret;
    }

    Tensor log10(const Tensor& x) {
        Tensor ret(x.shape(), x.get_dtype(), x.get_device(), x.enable_grad());
        get_log10_kernel(x.get_device())(x, ret);
        if (DLEngine::is_grad_mode()) {
            ret.grad_func() = std::make_shared<LogLayer>(x.get_dtype(), x.get_device(), false);
            ret.grad_func()->add_input(x);
            // Add the base 10 for the log
            Tensor base({1}, x.get_dtype(), x.get_device(), x.enable_grad());
            base.fill_(10);
            ret.grad_func()->add_input(base);
            ret.grad_func()->add_output(ret);
        }

        return ret;
    }

    Tensor sin(const Tensor& x) {
        Tensor ret(x.shape(), x.get_dtype(), x.get_device(), x.enable_grad());
        get_sin_kernel(x.get_device())(x, ret);
        if (DLEngine::is_grad_mode()) {
            ret.grad_func() = std::make_shared<SinLayer>(x.get_dtype(), x.get_device(), false);
            ret.grad_func()->add_input(x);
            ret.grad_func()->add_output(ret);
        }

        return ret;
    }

    Tensor cos(const Tensor& x) {
        Tensor ret(x.shape(), x.get_dtype(), x.get_device(), x.enable_grad());
        get_cos_kernel(x.get_device())(x, ret);
        if (DLEngine::is_grad_mode()) {
            ret.grad_func() = std::make_shared<CosLayer>(x.get_dtype(), x.get_device(), false);
            ret.grad_func()->add_input(x);
            ret.grad_func()->add_output(ret);
        }

        return ret;
    }

    Tensor tan(const Tensor& x) {
        Tensor ret(x.shape(), x.get_dtype(), x.get_device(), x.enable_grad());
        get_tan_kernel(x.get_device())(x, ret);
        if (DLEngine::is_grad_mode()) {
            ret.grad_func() = std::make_shared<TanLayer>(x.get_dtype(), x.get_device(), false);
            ret.grad_func()->add_input(x);
            ret.grad_func()->add_output(ret);
        }

        return ret;
    }

}
