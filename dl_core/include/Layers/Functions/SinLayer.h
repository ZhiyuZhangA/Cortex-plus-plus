#ifndef SIN_LAYER_H
#define SIN_LAYER_H
#include "../BaseLayer.h"

namespace cortex {
    class SinLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the sin layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit SinLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit SinLayer(bool supportQuantization);

        ~SinLayer() override = default;

        void backward() override;
    };
}

#endif
