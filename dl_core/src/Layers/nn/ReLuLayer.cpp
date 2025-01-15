#include "Layers/nn/ReLuLayer.h"

#include "Layers/Kernels/DeviceKernel.h"
#include "Tensor/Tensor.h"

namespace cortex {
    ReLuLayer::ReLuLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "ReLu Layer";
    }

    ReLuLayer::ReLuLayer(const bool supportQuantization) : ReLuLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void ReLuLayer::backward() {
        if (this->m_inputs[0].enable_grad()) {
            *(this->m_inputs[0].grad()) += get_drelu_kernel(m_deviceType)(this->m_inputs[0]) * *(this->m_outputs[0].grad());
        }
    }
}