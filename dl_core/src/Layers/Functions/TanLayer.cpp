#include "Layers/Functions/TanLayer.h"
#include "Functions/math_utils.h"
#include "Tensor/Tensor.h"

namespace cortex {
    TanLayer::TanLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Tan Layer";
    }

    TanLayer::TanLayer(const bool supportQuantization) : TanLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void TanLayer::backward() {
        if (this->m_inputs[0].enable_grad()) {
            *(this->m_inputs[0].grad()) += (1.0f / cos(this->m_inputs[0])) * (1.0f / cos(this->m_inputs[0])) * *(this->m_outputs[0].grad());
        }
    }


}