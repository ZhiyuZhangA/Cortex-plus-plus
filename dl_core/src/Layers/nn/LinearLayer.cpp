#include "Layers/nn/LinearLayer.h"
#include "Tensor/Tensor.h"

namespace cortex {
    LinearLayer::LinearLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Linear Layer";
    }

    LinearLayer::LinearLayer(bool supportQuantization) : LinearLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void LinearLayer::backward() {

    }


}