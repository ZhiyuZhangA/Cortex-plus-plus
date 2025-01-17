#ifndef MSE_LOSS_LAYER_H
#define MSE_LOSS_LAYER_H
#include "Layers/BaseLayer.h"

namespace cortex {
    class MSELossLayer final : public BaseLayer {
    public:
        /**
        * Constructor of the mse loss with data type and device type specified
        * @param dtype
        * @param deviceType
        * @param supportQuantization
        */
        explicit MSELossLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit MSELossLayer(bool supportQuantization);

        ~MSELossLayer() override = default;

        void backward() override;
    };
}

#endif
