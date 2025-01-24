#include "Layers/nn/CrossEntropyLayer.h"

#include "Functions/nn_utils.h"
#include "Layers/Kernels/DeviceKernel.h"

namespace cortex {

    CrossEntropyLayer::CrossEntropyLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) : BaseLayer(dtype, deviceType, supportQuantization) {
        this->m_layerName = "Cross Entropy Loss";
    }

    CrossEntropyLayer::CrossEntropyLayer(const bool supportQuantization) : CrossEntropyLayer(dtype::f32, DeviceType::cpu, supportQuantization) { }

    void CrossEntropyLayer::backward() {
        if (this->m_inputs[1].enable_grad()) {
            *(this->m_inputs[1].grad()) += (FSoftmax(this->m_inputs[1]) - this->m_inputs[0]) * *(this->m_outputs[0].grad());
        }
    }

}
