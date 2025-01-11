#include <iostream>
#include <Tensor/Tensor.h>
#include "Layers/BaseLayer.h"
#include "Random/RandomEngine.h"

using namespace dl_core;

int main() {

    const Tensor tensorA({1, 3, 2}, dtype::f32);
    tensorA.initialize_with({7, 8, 9, 2, 11, 100});
    const Tensor tensorB({2, 3, 2}, dtype::f32);
    tensorB.initialize_with({1, 2, 30, 4, 10, 6, 7, 8, 9, 2, 11, 100});
    std::cout << tensorB.to_string() << std::endl;

    std::cout << tensorA.broadcast_to({2, 3, 2}).to_string() << std::endl;

    const Tensor res = tensorA + tensorB;

    std::cout << res.to_string() << std::endl;

    return 0;
}
