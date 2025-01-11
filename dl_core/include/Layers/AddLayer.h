#ifndef ADD_LAYER_H
#define ADD_LAYER_H
#include "BaseLayer.h"

namespace dl_core {
    class AddLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the add layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit AddLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit AddLayer(bool supportQuantization);

        ~AddLayer() override = default;

        void backward() override;
    };
}

#endif
