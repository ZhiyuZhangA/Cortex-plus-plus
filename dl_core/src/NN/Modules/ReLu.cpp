#include "NN/Modules/ReLu.h"

#include <iostream>

#include "Functions/nn_utils.h"

namespace cortex {

    ReLu::ReLu(const dtype& dtype, const DeviceType& device)
            : BaseModule(dtype, device) {  }

    Tensor ReLu::forward(const Tensor& input) {
        return FReLu(input);
    }
}
