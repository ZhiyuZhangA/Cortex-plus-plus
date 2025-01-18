#include "Layers/nn/LinearLayer.h"

#include <iostream>

#include "Tensor/Tensor.h"

namespace cortex {
    LinearLayer::LinearLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Linear Layer";
    }

    LinearLayer::LinearLayer(const bool supportQuantization) : LinearLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void LinearLayer::backward() {
        // Compute the gradient for input and weight
        if (m_inputs[0].enable_grad()) {
            *(m_inputs[0].grad()) += m_outputs[0].grad()->matmul(m_inputs[1]);
        }

        if (m_inputs[1].enable_grad()) {
            std::vector<uint32_t> shape(m_outputs[0].grad()->shape().size(), 1);
            shape[shape.size() - 2] = m_inputs[1].shape()[m_inputs[1].dim() - 2];
            shape[shape.size() - 1] = m_inputs[1].shape()[m_inputs[1].dim() - 1];



            *(m_inputs[1].grad()) += m_outputs[0].grad()->transpose(m_outputs[0].dim() - 1, m_outputs[0].dim() - 2).matmul(m_inputs[0]).sum_to(shape);
        }



        // Bias term exists
        if (m_inputs.size() == 3) {
            std::vector<uint32_t> shape(m_outputs[0].grad()->shape().size(), 1);
            shape[shape.size() - 1] = m_outputs[0].shape()[m_outputs[0].dim() - 1];

            *(m_inputs[2].grad()) += m_outputs[0].grad()->sum_to(shape);
        }
    }
}