#ifndef LOSS_H
#define LOSS_H
#include "Tensor/Tensor.h"

namespace cortex {
    /**
     *
     * @param label
     * @param prediction
     * @param mode the reduction mode of mse loss
     *        - 0: None reduction, shape (prediction.shape)
     *        - 1: Sum, returns a scalar (summing all elements in the tensor)
     *        - 2: Mean, returns a scalar
     * @return
     */
    Tensor FMSELoss(const Tensor& label, const Tensor& prediction, uint8_t mode = 0);
}

#endif //LOSS_H
