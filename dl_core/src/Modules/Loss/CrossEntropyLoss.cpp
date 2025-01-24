#include "Modules/Loss/CrossEntropyLoss.h"
#include "Functions/loss.h"

namespace cortex {

    CrossEntropyLoss::CrossEntropyLoss(const dtype& dtype, const DeviceType& device)
            : BaseLoss(dtype, device) { }

    Tensor CrossEntropyLoss::forward(const Tensor& label, const Tensor& prediction) {
        return FCrossEntropyLoss(label, prediction);
    }
}
