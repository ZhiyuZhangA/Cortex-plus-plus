#ifndef BASE_LOSS_H
#define BASE_LOSS_H

#include "Dtypes/Dtype.h"
#include "Tensor/Tensor.h"

namespace cortex {
    class BaseLoss {
    protected:
        ~BaseLoss() = default;

    public:
        BaseLoss(const dtype dtype, const DeviceType device) { }

        virtual Tensor forward(const Tensor& label, const Tensor& prediction) = 0;

    };
}

#endif
