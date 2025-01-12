#ifndef SUBLAYER_H
#define SUBLAYER_H

#include "../BaseLayer.h"

namespace cortex {
    class SubLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the subtraction layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit SubLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit SubLayer(bool supportQuantization);

        ~SubLayer() override = default;

        void backward() override;
    };
}

#endif //SUBLAYER_H
