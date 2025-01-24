#ifndef CROSS_ENTROPY_LOSS_H
#define CROSS_ENTROPY_LOSS_H

#include "BaseLoss.h"
#include "Tensor/Tensor.h"

namespace cortex {

    class CrossEntropyLoss final : public BaseLoss {
    public:
        explicit CrossEntropyLoss(const dtype& dtype, const DeviceType& devices);

        Tensor forward(const Tensor& label, const Tensor& prediction) override;
    };

}

#endif
