#include "Layers/BaseLayer.h"

namespace dl_core {
    BaseLayer::BaseLayer(const dtype dtype, const DeviceType deviceType, const bool supportQuantization) {
        this->m_dtype = dtype;
        this->m_deviceType = deviceType;
        this->m_supportQuantization = supportQuantization;
    }

    void BaseLayer::add_input(const Tensor& input) {
        this->m_inputs.push_back(input);
    }

    void BaseLayer::add_output(const Tensor& output) {
        this->m_outputs.push_back(output);
    }
}
