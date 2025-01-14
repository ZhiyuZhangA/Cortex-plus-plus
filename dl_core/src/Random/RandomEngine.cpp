#include "Random/RandomEngine.h"

namespace cortex {

    RandomEngine::RandomEngine() {
        std::random_device rd;
        gen = std::mt19937(rd());
    }

    Tensor RandomEngine::uniform(const std::vector<ui32_t> &dim) {
        std::uniform_real_distribution<f32_t> dist;
        return generate_tensor<f32_t>(dim, dist);
    }

    Tensor RandomEngine::uniform(const std::vector<ui32_t> &dim, const f32_t &a, const f32_t &b) {
        std::uniform_real_distribution<f32_t> dist(a, b);
        return generate_tensor<f32_t>(dim, dist);
    }

    Tensor RandomEngine::normal(const std::vector<ui32_t> &dim) {
        std::normal_distribution<f32_t> dist;
        return generate_tensor<f32_t>(dim, dist);
    }

    Tensor RandomEngine::xavier_normal(const std::vector<ui32_t> &dim) {
        int fan_in = dim[dim.size() - 1];
        int fan_out = dim[dim.size() - 2];

        float stddev = std::sqrt(2.0 / (fan_in + fan_out));
        std::normal_distribution<f32_t> dist(0, stddev);
        return generate_tensor<f32_t>(dim, dist);
    }

    Tensor RandomEngine::kaiming_normal(const std::vector<ui32_t> &dim, const f32_t a) {
        const int fan_in = dim[dim.size() - 1];
        float stddev = std::sqrt(2.0 / ((1 + a * a) * fan_in));
        std::normal_distribution<f32_t> dist(0, stddev);
        return generate_tensor<f32_t>(dim, dist);
    }

    Tensor RandomEngine::kaiming_uniform(const std::vector<ui32_t> &dim, const f32_t a) {
        const int fan_in = dim[dim.size() - 1];
        float stddev = std::sqrt(6.0 / ((1 + a * a) * fan_in));
        std::normal_distribution<f32_t> dist(0, stddev);
        return generate_tensor<f32_t>(dim, dist);
    }

    Tensor RandomEngine::bernoulli(const std::vector<ui32_t> &dim, float prob) {
        std::bernoulli_distribution dist(prob);
        return generate_tensor<i32_t>(dim, dist);
    }
}