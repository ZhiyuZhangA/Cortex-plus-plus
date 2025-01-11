#ifndef MUL_LAYER_H
#define MUL_LAYER_H
#include "BaseLayer.h"

namespace cortex_core {
    class MulLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the multiplication layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit MulLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit MulLayer(bool supportQuantization);

        ~MulLayer() override = default;

        void backward() override;
    };
}


#endif //MULLAYER_H
