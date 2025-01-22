#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H
#include "Modules/BaseModule.h"

namespace cortex {
    class Sequential final : public BaseModule {
    public:
        Sequential(const std::vector<std::shared_ptr<BaseModule>>& modules, const dtype dtype, const DeviceType device)
            : BaseModule(dtype, device), m_modules(modules) {
            // Add Params
            for (const auto& module_ptr : modules) {
                for (auto & param : module_ptr->get_params()) {
                    m_params.push_back(param);
                }
            }
        }

        Tensor forward(const Tensor &input) override {
            if (m_modules.empty()) {
                throw std::invalid_argument("Sequential: modules is empty");
            }

            Tensor ret = m_modules[0]->forward(input);
            for (int i = 1; i < m_modules.size(); ++i) {
                ret = m_modules[i]->forward(ret);
            }
            return ret;
        }

    private:
        std::vector<std::shared_ptr<BaseModule>> m_modules;
    };
}

#endif //SEQUENTIAL_H
