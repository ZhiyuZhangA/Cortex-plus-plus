#ifndef BASE_LAYER_H
#define BASE_LAYER_H

#include <vector>
#include <array>
#include "DeviceAllocator/DeviceAllocator.h"
#include "Dtypes/Dtype.h"
#include "Tensor/Tensor.h"

namespace cortex {

    class BaseLayer : std::enable_shared_from_this<BaseLayer> {
    public:
        explicit BaseLayer(dtype dtype, DeviceType deviceType, bool supportQuantization);
        virtual ~BaseLayer() = default;

        virtual void backward() = 0;

        void add_input(const Tensor& input);
        void add_output(const Tensor& output);
        void add_param(const float& param);

        std::vector<Tensor> get_inputs() const;

    protected:
        std::string m_layerName;
        dtype m_dtype = dtype::None;
        DeviceType m_deviceType = DeviceType::Unknown;
        bool m_supportQuantization = false;

        std::vector<Tensor> m_inputs;
        std::vector<Tensor> m_outputs;
        std::vector<float> m_params;
    };

}

#endif
