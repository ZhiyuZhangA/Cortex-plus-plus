#ifndef POW_LAYER_H
#define POW_LAYER_H
#include "../BaseLayer.h"

namespace cortex {
    class PowLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the power layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit PowLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit PowLayer(bool supportQuantization);

        ~PowLayer() override = default;

        void backward() override;
    };
}

#endif
