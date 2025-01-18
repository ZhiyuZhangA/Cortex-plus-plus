#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include <memory>

#include "Tensor/Tensor.h"

namespace cortex {
    class BaseOptimizer {
    public:
        virtual ~BaseOptimizer() = default;

        explicit BaseOptimizer(const std::vector<Tensor*>& parameters, const float& lr = 0.01f) : m_lr(lr) {
            for (const auto& parameter : parameters) {
                m_parameters.emplace_back(parameter);
            }
        }

        virtual void step() = 0;

        void zero_grads() const {
            for (auto param : m_parameters) {
                param->zero_grad();
            }
        }

    protected:
        std::shared_ptr<Autograd_graph> m_graph;
        std::vector<Tensor*> m_parameters;
        float m_lr;
    };

    class SGD final : public BaseOptimizer {
    public:
        SGD(const std::vector<Tensor*> &parameters, const float &lr)
            : BaseOptimizer(parameters, lr) { }

        void step() override {
            for (const auto& param : m_parameters) {
                (*param) -= m_lr * *(param->grad());
            }
        }
    };
}

#endif //OPTIMIZER_H
