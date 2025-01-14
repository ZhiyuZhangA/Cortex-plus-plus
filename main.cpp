#include <iostream>
#include <Tensor/Tensor.h>

#include "Functions/nn_utils.h"
#include "Layers/BaseLayer.h"
#include "NN/Modules/Linear.h"
#include "Random/RandomEngine.h"

using namespace cortex;

int main() {

    RandomEngine rng;
    // const auto tensorA = rng.normal({2, 3, 2});
    // const auto tensorB = rng.normal({2, 2, 4});

    const auto tensorA = Tensor({2, 3, 2}, dtype::f32, DeviceType::cpu, true);
    // const auto tensorB = Tensor({2, 4, 2}, dtype::f32, DeviceType::cpu, true);
    // const auto bias = Tensor({1, 4}, dtype::f32, DeviceType::cpu, true);
    // tensorA.initialize_with({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    // tensorB.initialize_with({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
    // bias.fill_(3.0);

    Linear linear(dtype::f32, DeviceType::cpu, 2, 4, true);
    auto res = linear.forward(tensorA);

    std::cout << res.to_string() << std::endl;

    return 0;
}
