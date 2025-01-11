#ifndef ATAN_LAYER_H
#define ATAN_LAYER_H
#include "BaseLayer.h"

namespace cortex {

    class ATanLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the atan layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit ATanLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit ATanLayer(bool supportQuantization);

        ~ATanLayer() override = default;

        void backward() override;
    };
}


#endif