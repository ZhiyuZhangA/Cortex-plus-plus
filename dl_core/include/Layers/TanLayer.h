#ifndef TAN_LAYER_H
#define TAN_LAYER_H
#include "BaseLayer.h"

namespace cortex {
    class TanLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the tan layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit TanLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit TanLayer(bool supportQuantization);

        ~TanLayer() override = default;

        void backward() override;
    };
}



#endif
