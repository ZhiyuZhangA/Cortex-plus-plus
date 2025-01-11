#ifndef BROADCAST_LAYER_H
#define BROADCAST_LAYER_H

#include "BaseLayer.h"

namespace cortex {
    class BroadcastLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the broadcast layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit BroadcastLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit BroadcastLayer(bool supportQuantization);

        ~BroadcastLayer() override = default;

        void backward() override;
    };
}

#endif
