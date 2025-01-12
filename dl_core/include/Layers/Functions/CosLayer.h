#ifndef COS_LAYER_H
#define COS_LAYER_H
#include "../BaseLayer.h"

namespace cortex {
    class CosLayer final : public BaseLayer {
    public:
        /**
         * Constructor of the cos layer with data type and device type specified
         * @param dtype
         * @param deviceType
         * @param supportQuantization
         */
        explicit CosLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);

        /**
         * Constructor
         * @param supportQuantization support Quantization or not
         */
        explicit CosLayer(bool supportQuantization);

        ~CosLayer() override = default;

        void backward() override;
    };
}



#endif
