#include "Layers/Functions/ExpLayer.h"
#include "Tensor/Tensor.h"
#include "Functions/math_utils.h"

namespace cortex {
    ExpLayer::ExpLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Exp Layer";
    }

    ExpLayer::ExpLayer(bool supportQuantization) : ExpLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void ExpLayer::backward() {
        if (this->m_inputs[0].enable_grad()) {
            *(this->m_inputs[0].grad()) += exp(this->m_inputs[0]) * *(this->m_outputs[0].grad());
        }
    }


}