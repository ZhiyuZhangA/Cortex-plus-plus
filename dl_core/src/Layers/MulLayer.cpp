#include "Layers/MulLayer.h"
#include "Layers/Kernels/DeviceKernel.h"
#include "Tensor/Tensor.h"

namespace dl_core {
    MulLayer::MulLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Mul Layer";
    }

    MulLayer::MulLayer(bool supportQuantization) : MulLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void MulLayer::backward() {
        if (m_inputs[0].enable_grad())
            *(m_inputs[0].grad()) += m_inputs[1] * *(m_outputs[0].grad());

        if (m_inputs[1].enable_grad())
            *(m_inputs[1].grad()) += m_inputs[0] * *(m_outputs[0].grad());
    }


}