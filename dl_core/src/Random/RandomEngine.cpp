#include "Random/RandomEngine.h"

namespace cortex {

    RandomEngine::RandomEngine() {
        std::random_device rd;
        gen = std::mt19937(rd());
    }

    Tensor RandomEngine::bernoulli(const std::vector<ui32_t> &dim, float prob) {
        std::bernoulli_distribution dist(prob);
        return generate_tensor<i32_t>(dim, dist);
    }
}