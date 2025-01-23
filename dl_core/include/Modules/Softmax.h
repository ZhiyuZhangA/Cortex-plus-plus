#ifndef SOFTMAX_H
#define SOFTMAX_H
#include "BaseModule.h"

namespace cortex {

    class Softmax final : public BaseModule {
    public:
        explicit Softmax(const dtype& dtype, const DeviceType& device);

        Tensor forward(const Tensor& input) override;

    };

}


#endif
