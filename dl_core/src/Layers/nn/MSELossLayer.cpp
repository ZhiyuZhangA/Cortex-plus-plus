#include "Layers/nn/MSELossLayer.h"

namespace cortex {

    MSELossLayer::MSELossLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "MSE Loss Layer";
    }

    MSELossLayer::MSELossLayer(const bool supportQuantization) : MSELossLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void MSELossLayer::backward() {
        // Only calculate the gradient of prediction
        if (this->m_inputs[1].enable_grad()) {
            *(this->m_inputs[1].grad()) -= 2 * (this->m_inputs[0] - this->m_inputs[1]) * this->m_outputs[0].grad()->broadcast_to(this->m_inputs[0].shape());
            if (this->m_params[0] == 2)
                *(this->m_inputs[1].grad()) = *(this->m_inputs[1].grad()) / this->m_inputs[1].size();
        }
    }

}