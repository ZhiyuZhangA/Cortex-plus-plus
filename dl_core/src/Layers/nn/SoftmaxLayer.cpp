#include "Layers/nn/SoftmaxLayer.h"

#include <iostream>

#include "Layers/Kernels/DeviceKernel.h"

namespace cortex {

    SoftmaxLayer::SoftmaxLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Softmax Layer";
    }

    SoftmaxLayer::SoftmaxLayer(const bool supportQuantization) : SoftmaxLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void SoftmaxLayer::backward() {
        if (this->m_inputs[0].enable_grad()) {
            *(this->m_inputs[0].grad()) += get_dsoftmax_kernel(this->m_deviceType)(this->m_outputs[0]) * *(this->m_outputs[0].grad());
        }
    }

}
