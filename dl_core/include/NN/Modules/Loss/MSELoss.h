#ifndef MSE_LOSS_H
#define MSE_LOSS_H

#include "BaseLoss.h"
#include "Tensor/Tensor.h"

namespace cortex {

    class MSELoss final : public BaseLoss {
    public:
        explicit MSELoss(const dtype& dtype, const DeviceType& devices, uint8_t mode = 2);

        Tensor forward(const Tensor& label, const Tensor& prediction) override;

    private:
        uint8_t m_mode;
    };

}


#endif
