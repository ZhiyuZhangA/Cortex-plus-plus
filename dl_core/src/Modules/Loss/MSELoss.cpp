#include "Modules/Loss/MSELoss.h"
#include "Functions/loss.h"


namespace cortex {

    MSELoss::MSELoss(const dtype& dtype, const DeviceType& device, uint8_t mode)
            : BaseLoss(dtype, device), m_mode(mode) { }

    Tensor MSELoss::forward(const Tensor& label, const Tensor& prediction) {
        return FMSELoss(label, prediction, m_mode);
    }
}
