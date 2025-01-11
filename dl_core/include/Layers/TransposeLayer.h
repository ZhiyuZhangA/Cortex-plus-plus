#ifndef TRANSPOSE_LAYER_H
#define TRANSPOSE_LAYER_H

#include "BaseLayer.h"

namespace cortex {
    class TransposeLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the transpose layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit TransposeLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit TransposeLayer(bool supportQuantization);

        ~TransposeLayer() override = default;

        void backward() override;
    };
}


#endif
