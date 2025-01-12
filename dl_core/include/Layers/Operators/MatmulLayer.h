#ifndef MATMUL_LAYER_H
#define MATMUL_LAYER_H

#include "../BaseLayer.h"

namespace cortex {
    class MatmulLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the matmul layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit MatmulLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit MatmulLayer(bool supportQuantization);

        ~MatmulLayer() override = default;

        void backward() override;
    };
}

#endif
