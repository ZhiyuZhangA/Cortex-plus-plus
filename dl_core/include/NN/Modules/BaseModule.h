#ifndef BASE_MODULE_H
#define BASE_MODULE_H
#include "Random/RandomEngine.h"
#include "Tensor/Tensor.h"

namespace cortex {

    class BaseModule {
    public:
        BaseModule(const dtype dtype, const DeviceType device) : m_dtype(dtype), m_device(device) {};
        virtual Tensor forward(const Tensor& input) = 0;

    protected:
        ~BaseModule() = default;

        std::vector<Tensor> m_params;
        dtype m_dtype = dtype::f32;
        DeviceType m_device = DeviceType::cpu;
        RandomEngine m_randomEngine;
    };

}


#endif
