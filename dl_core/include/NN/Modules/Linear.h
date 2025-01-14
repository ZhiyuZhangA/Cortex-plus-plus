#ifndef LINEAR_H
#define LINEAR_H
#include "BaseModule.h"

namespace cortex {

    class Linear final : public BaseModule {
    public:
        explicit Linear(const dtype& dtype, const DeviceType& device, uint32_t in_features, uint32_t out_features, const bool& bias=true);

        Tensor forward(const Tensor& input) override;

        Tensor get_weight() const;

        Tensor get_bias() const;

    private:
        const bool& m_bias;
    };

}

#endif //LINEAR_H
