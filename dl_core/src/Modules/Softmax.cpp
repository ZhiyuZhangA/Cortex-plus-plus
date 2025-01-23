#include "Modules/Softmax.h"
#include "Functions/nn_utils.h"

namespace cortex {

    Softmax::Softmax(const dtype& dtype, const DeviceType& device)
            : BaseModule(dtype, device) {  }

    Tensor Softmax::forward(const Tensor& input) {
        return FSoftmax(input);
    }

}
