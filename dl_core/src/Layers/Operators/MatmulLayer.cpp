#include "Layers/Operators/MatmulLayer.h"
#include "Tensor/Tensor.h"

namespace cortex {
    MatmulLayer::MatmulLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Matmul Layer";
    }

    MatmulLayer::MatmulLayer(bool supportQuantization) : MatmulLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void MatmulLayer::backward() {
        if (this->m_inputs[0].enable_grad())
            *(this->m_inputs[0].grad()) += (this->m_outputs[0].grad())->matmul(this->m_inputs[1].transpose(this->m_inputs[1].dim() - 2, this->m_inputs[1].dim() - 1));

        if (this->m_inputs[1].enable_grad()) {
            *(this->m_inputs[1].grad()) += this->m_inputs[0].transpose(this->m_inputs[0].shape().size() - 2, this->m_inputs[0].shape().size() - 1).matmul(*(this->m_outputs[0].grad()));
        }
    }


}