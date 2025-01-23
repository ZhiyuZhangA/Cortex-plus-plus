#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H
#include "Layers/BaseLayer.h"

namespace cortex {
    class SoftmaxLayer final : public BaseLayer {
    public:
        /**
        * Constructor of the softmax with data type and device type specified
        * @param dtype
        * @param deviceType
        * @param supportQuantization
        */
        explicit SoftmaxLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit SoftmaxLayer(bool supportQuantization);

        ~SoftmaxLayer() override = default;

        void backward() override;
    };
}


#endif
