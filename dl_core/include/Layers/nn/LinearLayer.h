#ifndef LINEAR_LAYER_H
#define LINEAR_LAYER_H

#include "../BaseLayer.h"

namespace cortex {
    class LinearLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the linear layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit LinearLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit LinearLayer(bool supportQuantization);

        ~LinearLayer() override = default;

        void backward() override;
    };
}

#endif
