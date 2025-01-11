#ifndef RANDOMENGINE_H
#define RANDOMENGINE_H
#include <random>

#include "Dtypes/Dtype.h"
#include "Tensor/Tensor.h"

namespace dl_core {
    class RandomEngine {
    public:
        /**
         * Constructor of RandomEngine Class with specified random seed.
         * @param seed the random seed of the RandomEngine
         */
        explicit RandomEngine(unsigned seed) : gen(seed) { }

        /**
         * Default constructor of RandomEngine Class using 'std::random_device'.
         */
        RandomEngine();

        /**
         * Generate uniformly distributed random values within [0, 1] both inclusive for tensor.
         * @tparam T data type of element in tensor (fp32 in default)
         * @param dim the dimension of the tensor
         * @return A tensor of type T (fp32 in default) filled with random numbers in the range [0, 1] from uniform distribution
         */
        template <typename T = f32_t>
        Tensor uniform(const std::vector<ui32_t>& dim);

        /**
         * Generate normally distributed random values within [0, 1] both inclusive for tensor.
         * @tparam T data type of element in tensor (fp32 in default)
         * @param dim the dimension of the tensor
         * @return A tensor of type T (fp32 in default) filled with random numbers in the range [0, 1] from normal distribution
         */
        template <typename T = f32_t>
        Tensor normal(const std::vector<ui32_t>& dim);

        /**
         * Generate values for a tensor from a Xavier normal distribution.
         * @note Xavier normal initialization (also known as Glorot normal initialization) is a weight
         * initialization technique that draws values from a normal distribution with a mean of 0
         * and a standard deviation of `sqrt(2 / (fan_in + fan_out))`.
         * @tparam T template data type of element in tensor (fp32 in default)
         * @param dim the dimension of the tensor
         * @return A tensor of type T (fp32 in default) filled with random numbers from Xavier normal distribution
         */
        template <typename T = f32_t>
        Tensor xavier_normal(const std::vector<ui32_t>& dim);

        /**
         * Generate values for a tensor from a Kaiming normal distribution.
         * @note The Kaiming normal initialization is commonly used is commonly used for layers with ReLU
         * activation function in deep neural networks.
         * @tparam T template data type of element in tensor (fp32 in default)
         * @param dim the dimension of the tensor
         * @param a A floating-point value that controls the variance of the initialization (0.0 in default)
         * @return A tensor of type T (fp32 in default) filled with random numbers from Kaiming normal distribution
         */
        template <typename T = f32_t>
        Tensor kaiming_normal(const std::vector<ui32_t>& dim, f32_t a = 0.0);

        /**
         * Generate values for a tensor from a Kaiming uniform distribution
         * @tparam T template data type of element in tensor (fp32 in default)
         * @param dim the dimension of the tensor
         * @param a A floating-point value that controls the variance of the initialization (0.0 in default)
         * @return A tensor of type T (fp32 in default) filled with random numbers from Kaiming uniform distribution
         */
        template <typename T = f32_t>
        Tensor kaiming_uniform(const std::vector<ui32_t>& dim, f32_t a = 0.0);

        /**
         * Generate random values from a Bernoulli distribution for a tensor.
         * @note Bernoulli distribution is a discrete probability distribution of a random variable
         * that takes the value 1 with probability 'p' and 0 with probability '1 - p'.
         * @param dim the dimension of the tensor
         * @param prob the probability to generate value 1
         * @return A tensor filled with random values from a Bernoulli distribution
         */
        Tensor bernoulli(const std::vector<ui32_t>& dim, float prob);

    private:
        /**
         * Generate a tensor object given the distribution.
         * @tparam T the template data type of element in the tensor
         * @tparam Distribution the distribution type used for random generation
         * @param dim the dimension of the tensor
         * @param dist the distribution used for generating random values
         * @param dtype data type of element in the tensor (fp32 in default)
         * @param device device type for the tensor (cpu in default)
         * @return A tensor filled with random values given specified distribution type
         */
        template <typename T, typename Distribution>
        Tensor generate_tensor(const std::vector<ui32_t>& dim, Distribution& dist, dtype dtype = dtype::f32, DeviceType device = DeviceType::cpu) {
            Tensor tensor(dim, dtype, device);
            T* ptr = tensor.ptr<T>();
            for (int i = 0; i < tensor.size(); i++) {
                *(ptr + i) = dist(gen);
            }

            return tensor;
        }

    private:
        std::mt19937 gen;
    };

    template<typename T>
    Tensor RandomEngine::uniform(const std::vector<ui32_t> &dim) {
        std::uniform_real_distribution<T> dist;
        return generate_tensor<T>(dim, dist);
    }

    template<typename T>
    Tensor RandomEngine::normal(const std::vector<ui32_t> &dim) {
        std::normal_distribution<T> dist;
        return generate_tensor<T>(dim, dist);
    }

    template<typename T>
    Tensor RandomEngine::xavier_normal(const std::vector<ui32_t> &dim) {
        int fan_in = dim[dim.size() - 1];
        int fan_out = dim[dim.size() - 2];

        float stddev = std::sqrt(2.0 / (fan_in + fan_out));
        std::normal_distribution<T> dist(0, stddev);
        return generate_tensor<T>(dim, dist);
    }

    template<typename T>
    Tensor RandomEngine::kaiming_normal(const std::vector<ui32_t> &dim, const f32_t a) {
        const int fan_in = dim[dim.size() - 1];
        float stddev = std::sqrt(2.0 / ((1 + a * a) * fan_in));
        std::normal_distribution<T> dist(0, stddev);
        return generate_tensor<T>(dim, dist);
    }

    template<typename T>
    Tensor RandomEngine::kaiming_uniform(const std::vector<ui32_t> &dim, const f32_t a) {
        const int fan_in = dim[dim.size() - 1];
        float stddev = std::sqrt(6.0 / ((1 + a * a) * fan_in));
        std::normal_distribution<T> dist(0, stddev);
        return generate_tensor<T>(dim, dist);
    }
}


#endif
