#include "NN/Modules/ReLu.h"
#include "Functions/nn_utils.h"

namespace cortex {

    ReLu::ReLu(const dtype& dtype, const DeviceType& device)
            : BaseModule(dtype, device) {  }

    Tensor ReLu::forward(const Tensor& input) {
        return FReLu(input);
    }

    LeakyReLu::LeakyReLu(const dtype &dtype, const DeviceType &device, const float& slope) : BaseModule(dtype, device), m_slope(slope) { }

    Tensor LeakyReLu::forward(const Tensor &input) {
        return FLeakyReLu(input, m_slope);
    }
}
