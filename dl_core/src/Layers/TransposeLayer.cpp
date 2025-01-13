#include "Layers/TransposeLayer.h"
#include "Tensor/Tensor.h"

namespace cortex {
    TransposeLayer::TransposeLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Transpose Layer";
    }

    TransposeLayer::TransposeLayer(bool supportQuantization) : TransposeLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void TransposeLayer::backward() {
        if (m_inputs[0].enable_grad()) {
            uint32_t dim0 = *(m_inputs[1].ptr<float>()), dim1 = *(m_inputs[2].ptr<float>());
            *(m_inputs[0].grad()) += m_outputs[0].grad()->transpose(dim0, dim1);
        }

    }

}