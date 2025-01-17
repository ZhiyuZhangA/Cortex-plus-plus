#ifndef SIGMOID_LAYER_H
#define SIGMOID_LAYER_H

#include "../BaseLayer.h"

namespace cortex {
    class SigmoidLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the sigmoid layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit SigmoidLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit SigmoidLayer(bool supportQuantization);

        ~SigmoidLayer() override = default;

        void backward() override;
    };
}

#endif
