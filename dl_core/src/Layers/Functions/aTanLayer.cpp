#include "Layers/Functions/aTanLayer.h"
#include "Tensor/Tensor.h"

namespace cortex {
    ATanLayer::ATanLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "aTan Layer";
    }

    ATanLayer::ATanLayer(const bool supportQuantization) : ATanLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void ATanLayer::backward() {
        if (this->m_inputs[0].enable_grad()) {
            *(this->m_inputs[0].grad()) += 1 / (1 + this->m_inputs[0] * this->m_inputs[0]) * *(this->m_outputs[0].grad());
        }
    }


}