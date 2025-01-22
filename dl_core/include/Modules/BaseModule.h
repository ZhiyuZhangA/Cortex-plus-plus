#ifndef BASE_MODULE_H
#define BASE_MODULE_H
#include "Random/RandomEngine.h"
#include "Tensor/Tensor.h"

namespace cortex {
    /**
     * Base Module is the base class for all layers except the loss function
     */
    class BaseModule {
    public:
        BaseModule(const dtype dtype, const DeviceType device) : m_dtype(dtype), m_device(device) {};

        /**
         * This function executes the forward propagation given the input tensor and return the results.
         * @param input the input tensor fed to forward propagation of current layer.
         * @return the output tensor.
         */
        virtual Tensor forward(const Tensor& input) = 0;

        /**
         * Get the parameters of current module
         * @return a vector of tensor representing the parameters of current module
         */
        std::vector<std::shared_ptr<Tensor>> get_params() const {
            return m_params;
        }

    protected:
        ~BaseModule() = default;

        std::vector<std::shared_ptr<Tensor>> m_params;
        dtype m_dtype = dtype::f32;
        DeviceType m_device = DeviceType::cpu;
        RandomEngine m_randomEngine;

    };

}


#endif
