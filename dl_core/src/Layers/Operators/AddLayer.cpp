#include "Layers/Operators/AddLayer.h"
#include "Layers/Kernels/DeviceKernel.h"
#include "Tensor/Tensor.h"

namespace cortex {
    AddLayer::AddLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Add Layer";
    }

    AddLayer::AddLayer(bool supportQuantization) : AddLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void AddLayer::backward() {
        if (m_inputs[0].enable_grad())
            *(m_inputs[0].grad()) += (m_outputs[0].grad())->sum_to(m_inputs[0].shape());

        if (m_inputs[1].enable_grad())
            *(m_inputs[1].grad()) += (m_outputs[0].grad())->sum_to(m_inputs[1].shape());
    }


}