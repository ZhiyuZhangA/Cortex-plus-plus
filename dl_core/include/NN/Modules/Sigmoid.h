#ifndef SIGMOID_H
#define SIGMOID_H
#include "BaseModule.h"

namespace cortex {

    class Sigmoid final : public BaseModule {
    public:
        explicit Sigmoid(const dtype& dtype, const DeviceType& device);

        Tensor forward(const Tensor& input) override;

    };

}


#endif
