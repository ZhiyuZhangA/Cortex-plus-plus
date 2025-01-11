#include "Layers/SubLayer.h"
#include "Tensor/Tensor.h"

namespace cortex {
    SubLayer::SubLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Sub Layer";
    }

    SubLayer::SubLayer(bool supportQuantization) : SubLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void SubLayer::backward() {
        if (m_inputs[0].enable_grad())
            *(m_inputs[0].grad()) -= *(m_outputs[0].grad());

        if (m_inputs[1].enable_grad())
            *(m_inputs[1].grad()) -= *(m_outputs[0].grad());
    }


}