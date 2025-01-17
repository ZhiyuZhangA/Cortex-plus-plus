#include "Layers/nn/SigmoidLayer.h"
#include "Tensor/Tensor.h"

namespace cortex {
    SigmoidLayer::SigmoidLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Sigmoid Layer";
    }

    SigmoidLayer::SigmoidLayer(const bool supportQuantization) : SigmoidLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void SigmoidLayer::backward() {

    }
}