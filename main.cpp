#include <iostream>
#include <Tensor/Tensor.h>
#include "Layers/BaseLayer.h"
#include "Random/RandomEngine.h"

using namespace cortex;

int main() {

    const Tensor tensorA({1, 3, 2}, dtype::f32);
    tensorA.initialize_with({7, 8, 9, 2, 11, 100});
    const Tensor tensorB({2, 3, 2}, dtype::f32);
    tensorB.initialize_with({1, 2, 30, 4, 10, 6, 7, 8, 9, 2, 11, 100});
    std::cout << tensorB.to_string() << std::endl;

    Tensor res = (tensorA + tensorB) * tensorB / 4;

    res.backward();

    std::cout << "A: " << tensorA.grad()->to_string() << std::endl;
    std::cout << "B: " << tensorB.grad()->to_string() << std::endl;


    return 0;
}
