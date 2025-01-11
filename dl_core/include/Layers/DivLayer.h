#ifndef DIV_LAYER_H
#define DIV_LAYER_H

#include "BaseLayer.h"

namespace cortex {
    class DivLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the division layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit DivLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit DivLayer(bool supportQuantization);

        ~DivLayer() override = default;

        void backward() override;
    };
}

#endif
