#ifndef AUTOGRAD_GRAPH_H
#define AUTOGRAD_GRAPH_H

#include <vector>
#include "Tensor/Tensor.h"

namespace cortex_core {

    class Autograd_graph {
    public:
        explicit Autograd_graph(const Tensor& output);

        void add_grad(const std::shared_ptr<Tensor>& grad);

        void backward(Tensor& output);

        void zero_grad() const;


    private:
        std::vector<std::shared_ptr<Tensor>> m_grads;
        Tensor m_output;
    };

}



#endif
