#ifndef SUM_TO_LAYER_H
#define SUM_TO_LAYER_H

#include "BaseLayer.h"

namespace cortex {
    class SumToLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the sum_to layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit SumToLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit SumToLayer(bool supportQuantization);

        ~SumToLayer() override = default;

        void backward() override;
    };
}

#endif
