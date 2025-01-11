#include "Layers/BroadcastLayer.h"
#include "Tensor/Tensor.h"

namespace cortex {
    BroadcastLayer::BroadcastLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Broadcast Layer";
    }

    BroadcastLayer::BroadcastLayer(bool supportQuantization) : BroadcastLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void BroadcastLayer::backward() {
        // Sum to
        if (this->m_inputs[0].enable_grad()) {
            *(this->m_inputs[0].grad()) += this->m_outputs[0].grad()->sum_to(this->m_inputs[0].shape());
        }
    }


}