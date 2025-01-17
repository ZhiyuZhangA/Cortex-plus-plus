#include "Functions/loss.h"
#include "DLEngine/DLEngine.h"
#include "Layers/BaseLayer.h"
#include "Layers/Kernels/DeviceKernel.h"
#include "Layers/nn/MSELossLayer.h"

namespace cortex {
    Tensor FMSELoss(const Tensor& label, const Tensor& prediction, const uint8_t mode) {
        Tensor ret({1}, prediction.get_dtype(), prediction.get_device(), true);
        get_mse_loss_kernel(prediction.get_device())(label, prediction, ret, mode);

        if (DLEngine::is_grad_mode()) {
            ret.grad_func() = std::make_shared<MSELossLayer>(prediction.get_dtype(), prediction.get_device(), false);
            ret.grad_func()->add_input(label);
            ret.grad_func()->add_input(prediction);
            ret.grad_func()->add_output(ret);
            ret.grad_func()->add_param(mode);
        }

        return ret;
    }

}