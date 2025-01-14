#include "NN/Modules/Linear.h"

#include <iostream>

#include "Functions/nn_utils.h"

namespace cortex {

    Linear::Linear(const dtype& dtype, const DeviceType& device, uint32_t in_features, uint32_t out_features, const bool& bias)
            : BaseModule(dtype, device), m_bias(bias) {
        // Initialize the weight parameter of size {out_features, in_features}
        const f32_t k = 1.0 / in_features;
        auto weight = m_randomEngine.uniform({out_features, in_features}, -std::sqrt(k), std::sqrt(k));
        weight.requires_grad();
        m_params.push_back(weight);

        // Initialize the bias parameter of size {1, out_features}
        if (bias) {
            auto bias_term = m_randomEngine.uniform({1, out_features}, -std::sqrt(k), std::sqrt(k));
            bias_term.requires_grad();
            m_params.push_back(bias_term);
        }
    }

    Tensor Linear::forward(const Tensor& input) {
        if (m_bias)
            return FLinear(input, m_params[0], m_params[1]);

        // Return no bias version
        return FLinear(input, m_params[0]);
    }

    Tensor Linear::get_weight() const {
        return m_params[0];
    }

    Tensor Linear::get_bias() const {
        if (!m_bias)
            throw std::runtime_error("No bias tensor provided in Linear Layer!");
        return m_params[1];
    }
}
