#include "NN/Modules/Sigmoid.h"
#include "Functions/nn_utils.h"

namespace cortex {

    Sigmoid::Sigmoid(const dtype& dtype, const DeviceType& device)
            : BaseModule(dtype, device) {  }

    Tensor Sigmoid::forward(const Tensor& input) {
        return FSigmoid(input);
    }

}
