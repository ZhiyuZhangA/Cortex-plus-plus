#include "Layers/TransposeLayer.h"
#include "Layers/Kernels/DeviceKernel.h"
#include "Tensor/Tensor.h"

namespace cortex_core {
    TransposeLayer::TransposeLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Transpose Layer";
    }

    TransposeLayer::TransposeLayer(bool supportQuantization) : TransposeLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void TransposeLayer::backward() {
        // if (m_inputs[0].enable_grad())
        //     m_inputs[0].grad()->transpose();
    }

}