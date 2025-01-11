#include "Layers/CosLayer.h"
#include "Tensor/math_utils.h"
#include "Tensor/Tensor.h"

namespace cortex {
    CosLayer::CosLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Cos Layer";
    }

    CosLayer::CosLayer(bool supportQuantization) : CosLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void CosLayer::backward() {
        if (this->m_inputs[0].enable_grad()) {
            *(this->m_inputs[0].grad()) -= sin(this->m_inputs[0]) * *(this->m_outputs[0].grad());
        }
    }


}