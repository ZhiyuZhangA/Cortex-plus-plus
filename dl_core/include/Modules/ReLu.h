#ifndef RELU_H
#define RELU_H
#include "BaseModule.h"

namespace cortex {
    class ReLu final: public BaseModule {
    public:
        explicit ReLu(const dtype& dtype, const DeviceType& device);

        Tensor forward(const Tensor& input) override;
    };

    class LeakyReLu final: public BaseModule {
    public:
        explicit LeakyReLu(const dtype& dtype, const DeviceType& device, const float& slope);

        Tensor forward(const Tensor& input) override;

        float get_slope() const { return m_slope; }

    private:
        float m_slope = 0.1f;
    };
}

#endif //RELU_H
