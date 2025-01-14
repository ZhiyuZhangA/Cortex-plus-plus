#include "Layers/nn/ReLuLayer.h"
#include "Tensor/Tensor.h"

namespace cortex {
    ReLuLayer::ReLuLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "ReLu Layer";
    }

    ReLuLayer::ReLuLayer(const bool supportQuantization) : ReLuLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void ReLuLayer::backward() {

    }
}