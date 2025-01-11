#ifndef EXP_LAYER_H
#define EXP_LAYER_H
#include "BaseLayer.h"

namespace cortex_core {
    class ExpLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the exponential layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit ExpLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit ExpLayer(bool supportQuantization);

        ~ExpLayer() override = default;

        void backward() override;
    };
}

#endif
