#ifndef BASEMODEL_H
#define BASEMODEL_H
#include <memory>
#include <vector>
#include "Tensor/Tensor.h"

namespace cortex {
    class BaseModel {
    public:
        explicit BaseModel();

    protected:
        std::vector<std::shared_ptr<Tensor>> m_parameters;
    };
}

#endif //BASEMODEL_H
