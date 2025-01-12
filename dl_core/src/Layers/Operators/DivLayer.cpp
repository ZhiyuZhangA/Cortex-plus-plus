#include "../../../include/Layers/Operators/DivLayer.h"
#include "../../../include/Tensor/Tensor.h"

namespace cortex {
    DivLayer::DivLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Div Layer";
    }

    DivLayer::DivLayer(bool supportQuantization) : DivLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void DivLayer::backward() {
        if (m_inputs[0].enable_grad())
            *(m_inputs[0].grad()) += 1.0f / m_inputs[1] * *(m_outputs[0].grad());

        if (m_inputs[1].enable_grad())
            *(m_inputs[1].grad()) -= m_inputs[0] * 1.0f / (m_inputs[1] * m_inputs[1]) * *(m_outputs[0].grad());
    }


}