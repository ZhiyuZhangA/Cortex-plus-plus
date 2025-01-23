#include "Layers/SumToLayer.h"

#include <iostream>

#include "Tensor/Tensor.h"

namespace cortex {
    SumToLayer::SumToLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "SumTo Layer";
    }

    SumToLayer::SumToLayer(bool supportQuantization) : SumToLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void SumToLayer::backward() {
        if (this->m_inputs[0].enable_grad()) {
            *(this->m_inputs[0].grad()) += this->m_outputs[0].grad()->broadcast_to(this->m_inputs[0].shape());
        }
    }


}