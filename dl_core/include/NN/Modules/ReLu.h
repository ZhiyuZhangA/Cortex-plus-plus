#ifndef RELU_H
#define RELU_H
#include "BaseModule.h"

namespace cortex {
    class ReLu final: public BaseModule {
    public:
        explicit ReLu(const dtype& dtype, const DeviceType& device);

        Tensor forward(const Tensor& input) override;
    };
}

#endif //RELU_H
