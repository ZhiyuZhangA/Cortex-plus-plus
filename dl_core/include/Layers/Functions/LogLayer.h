#ifndef LOG_LAYER_H
#define LOG_LAYER_H
#include "../BaseLayer.h"

namespace cortex {
    class LogLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the logarithmic layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit LogLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit LogLayer(bool supportQuantization);

        ~LogLayer() override = default;

        void backward() override;
    };
}

#endif
