#include "Layers/SinLayer.h"
#include "Tensor/math_utils.h"
#include "Tensor/Tensor.h"

namespace cortex_core {
    SinLayer::SinLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Sin Layer";
    }

    SinLayer::SinLayer(bool supportQuantization) : SinLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void SinLayer::backward() {
        if (this->m_inputs[0].enable_grad()) {
            *(this->m_inputs[0].grad()) += cos(this->m_inputs[0]) * *(this->m_outputs[0].grad());
        }
    }


}