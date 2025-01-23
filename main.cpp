#include <chrono>
#include <fstream>
#include <iostream>

#include "Functions/loss.h"
#include "Functions/math_utils.h"
#include "Functions/nn_utils.h"
#include "Tensor/Tensor.h"
#include "Layers/BaseLayer.h"
#include "Layers/Kernels/x86/nn_kernel_cpu.h"
#include "Models/Sequential.h"
#include "Modules/Linear.h"
#include "Modules/ReLu.h"
#include "Modules/Loss/MSELoss.h"
#include "Optimizers/BaseOptimizer.h"

using namespace cortex;

int main() {

    Tensor input({1, 12}, dtype::f32, DeviceType::cpu, true);
    // input.initialize_with({2.0, 1.0, 0.1, 0.5, 2.0, 1.0});
    input.initialize_with({-0.1875, 0.8026, 0.4352, 2.2096, 0.5503, 0.9852, 0.9854, 0.1866, -0.7313, -0.1727, -0.6427, 1.6189});

    Tensor res = cortex::exp(input) / cortex::exp(input).sum();
    Tensor output = FSoftmax(input);
    res.backward();
    std::cout << res.to_string() << std::endl;
    std::cout << input.grad()->to_string() << std::endl;

    return 0;
}
