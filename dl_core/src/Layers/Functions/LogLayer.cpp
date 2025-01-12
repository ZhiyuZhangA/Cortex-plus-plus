#include "Layers/Functions/LogLayer.h"
#include "Tensor/math_utils.h"
#include "Tensor/Tensor.h"

namespace cortex {
    LogLayer::LogLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Log Layer";
    }

    LogLayer::LogLayer(bool supportQuantization) : LogLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void LogLayer::backward() {
        // Two input, first one is the value, the other one is the base
        if (this->m_inputs[0].enable_grad()) {
            if (*(this->m_inputs[1].ptr<f32_t>()) == std::exp(1)) {
                *(this->m_inputs[0].grad()) += 1.0f / this->m_inputs[0] * *(this->m_outputs[0].grad());
            }
            else {
                *(this->m_inputs[0].grad()) += 1.0f / (this->m_inputs[0] * log(this->m_inputs[1])) * *(this->m_outputs[0].grad());
            }
        }

        if (this->m_inputs[1].enable_grad()) {
            *(this->m_inputs[1].grad()) -= log(this->m_inputs[1]) / (this->m_inputs[0] * log(this->m_inputs[1]) * log(this->m_inputs[1])) * *(this->m_outputs[0].grad());
        }
    }


}