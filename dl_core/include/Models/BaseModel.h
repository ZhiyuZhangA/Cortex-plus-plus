#ifndef BASEMODEL_H
#define BASEMODEL_H
#include <memory>
#include <vector>
#include "Tensor/Tensor.h"

namespace cortex {

    class BaseModel {
    public:
        virtual ~BaseModel() = default;

        explicit BaseModel() = default;

        BaseModel(const BaseModel&) = delete;

        /**
         * This function returns all parameters contained in the current model.
         * @return a std::vector containing the pointer pointing to tensor parameters
         */
        std::vector<std::shared_ptr<Tensor> > get_params() const {
            return m_parameters;
        }

        virtual Tensor forward(const Tensor& input) = 0;

    protected:
        std::vector<std::shared_ptr<Tensor>> m_parameters;
    };
}

#endif //BASEMODEL_H
