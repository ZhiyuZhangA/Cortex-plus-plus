#include <chrono>
#include <iostream>
#include "Tensor/Tensor.h"
#include "Layers/BaseLayer.h"
#include "NN/Modules/Linear.h"
#include "Random/RandomEngine.h"
#include "immintrin.h"
#include "avx2_extension/avx2_math_ext.h"
#include "NN/Modules/Loss/MSELoss.h"

using namespace cortex;



int main() {
    // const auto tensorA = Tensor({2, 3, 2}, dtype::f32, DeviceType::cpu, true);
    // tensorA.initialize_with({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    //
    // Linear linear(dtype::f32, DeviceType::cpu, 2, 4, true);
    // auto res = linear.forward(tensorA);
    // res.backward();
    //
    // std::cout << "Result:" << std::endl;
    // std::cout << res.to_string() << std::endl;
    // std::cout << "X.grad:" < std::endl;
    // std::cout << tensorA.grad()->to_string() << std::endl;
    // std::cout << "Weight.grad:" << std::endl;
    // std::cout << linear.get_weight().grad()->to_string() << std::endl;
    // std::cout << "Bias.grad:" << std::endl;
    // std::cout << linear.get_bias().grad()->to_string() << std::endl;

    Tensor label({2, 2}, dtype::f32 ,DeviceType::cpu, true);
    Tensor prediction({2, 2}, dtype::f32, DeviceType::cpu, true);

    label.fill_(2.3f);
    prediction.fill_(1.0f);

    MSELoss loss(dtype::f32, DeviceType::cpu, 2);
    auto l = loss.forward(label, prediction);

    l.backward();

    std::cout << prediction.grad()->to_string() << std::endl;

    return 0;
}
