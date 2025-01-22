#include <chrono>
#include <fstream>
#include <iostream>

#include "Functions/loss.h"
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

    RandomEngine random_engine;
    Tensor input = random_engine.normal({1, 12});
    std::cout << input.to_string() << std::endl;
    Tensor output({1, 12}, dtype::f32);

    softmax_kernel_cpu(input, output);

    std::cout << output.to_string() << std::endl;

    return 0;
}
