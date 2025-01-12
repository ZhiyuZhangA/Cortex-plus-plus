#include "Layers/Operators/MatmulLayer.h"
#include "Tensor/Tensor.h"

namespace cortex {
    MatmulLayer::MatmulLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Matmul Layer";
    }

    MatmulLayer::MatmulLayer(bool supportQuantization) : MatmulLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void MatmulLayer::backward() {

    }


}