#include <chrono>
#include <iostream>
#include "Tensor/Tensor.h"
#include "Layers/BaseLayer.h"
#include "NN/Modules/Linear.h"
#include "Random/RandomEngine.h"
#include "immintrin.h"
#include "avx2_extension/avx2_math_ext.h"

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
    // std::cout << "X.grad:" << std::endl;
    // std::cout << tensorA.grad()->to_string() << std::endl;
    // std::cout << "Weight.grad:" << std::endl;
    // std::cout << linear.get_weight().grad()->to_string() << std::endl;
    // std::cout << "Bias.grad:" << std::endl;
    // std::cout << linear.get_bias().grad()->to_string() << std::endl;

    float arr[8] = {0.4653, 0.7761, 0.7094, 0.5759, 0.1992, 0.7880, 0.4316, 0.6491};

    // __m256 vec = _mm256_load_ps(arr);
    // _mm256_store_ps(arr, _mm256_exp_ps(vec));

    for (int i = 0; i < 8; i++) {
        // std::cout << arr[i] << std::endl;
        std::cout << std::exp(arr[i]) << std::endl;
    }

    return 0;
}
