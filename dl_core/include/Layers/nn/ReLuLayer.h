#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "../BaseLayer.h"

namespace cortex {
    class ReLuLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the relu layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit ReLuLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit ReLuLayer(bool supportQuantization);

        ~ReLuLayer() override = default;

        void backward() override;
    };

    class LeakyReLuLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the leaky relu layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit LeakyReLuLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit LeakyReLuLayer(bool supportQuantization);

        ~LeakyReLuLayer() override = default;

        void backward() override;
    };
}

#endif