#include "Layers/PowLayer.h"

#include "Tensor/math_utils.h"
#include "Tensor/Tensor.h"

namespace dl_core {
    PowLayer::PowLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Pow Layer";
    }

    PowLayer::PowLayer(bool supportQuantization) : PowLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void PowLayer::backward() {
        if (this->m_inputs[0].enable_grad()) {
            *(this->m_inputs[0].grad()) += this->m_inputs[1] * pow(this->m_inputs[0], this->m_inputs[1] - 1.0f) * *(this->m_outputs[0].grad());
        }
    }


}