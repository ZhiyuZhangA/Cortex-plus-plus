#include "Graphs/autograd_graph.h"

#include <queue>

#include "Layers/BaseLayer.h"

namespace cortex {
    Autograd_graph::Autograd_graph(const Tensor& output): m_output(output) { }

    void Autograd_graph::add_grad(const std::shared_ptr<Tensor>& grad) {
        m_grads.push_back(grad);
    }

    void Autograd_graph::backward(Tensor& output) {
        if (!output.enable_grad())
            output.requires_grad();

        output.grad()->fill_(1.0);
        std::queue<Tensor> q_front;
        q_front.push(output);
        while (!q_front.empty()) {
            auto cur_node = q_front.front();
            q_front.pop();

            add_grad(cur_node.grad());
            // If leaf node
            if (cur_node.grad_func() == nullptr)
                continue;

            cur_node.grad_func()->backward();
            for (const auto& input : cur_node.grad_func()->get_inputs()) {
                q_front.push(input);
            }
        }
    }

    void Autograd_graph::zero_grad() const {
        for (auto &grad : m_grads) {
            grad->zero_grad();
        }
    }
}
