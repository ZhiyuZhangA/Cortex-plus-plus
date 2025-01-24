#ifndef CROSS_ENTROPY_LAYER_H
#define CROSS_ENTROPY_LAYER_H
#include "Layers/BaseLayer.h"

namespace cortex {
    class CrossEntropyLayer final : public BaseLayer {
    public:
        /**
        * Constructor of the cross entropy loss with data type and device type specified
        * @param dtype
        * @param deviceType
        * @param supportQuantization
        */
        explicit CrossEntropyLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit CrossEntropyLayer(bool supportQuantization);

        ~CrossEntropyLayer() override = default;

        void backward() override;
    };
}

#endif
