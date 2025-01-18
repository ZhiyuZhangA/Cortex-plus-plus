#ifndef RANDOMENGINE_H
#define RANDOMENGINE_H
#include <random>

#include "Dtypes/Dtype.h"
#include "Tensor/Tensor.h"

namespace cortex {
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
         * @param shape the dimension of the tensor
         * @return A tensor of type T (fp32 in default) filled with random numbers in the range [0, 1] from uniform distribution
         */
        Tensor uniform(const std::vector<ui32_t>& shape);

        /**
         * Generate uniformly distributed random values within [a, b) both inclusive for tensor.
         * @param shape the shape of the tensor
         * @param a the lower bound of the distribution (included)
         * @param b the upper bound of the distribution (not included)
         * @return A tensor of type (fp32 in default) filled with random numbers in the range [a, b) from uniform distribution
         */
        Tensor uniform(const std::vector<ui32_t>& shape, const f32_t& a, const f32_t& b);

        /**
         * Generate uniformly distributed random values within [a, b) both inclusive for the input tensor.
         * @param a the lower bound of the distribution (included)
         * @param b the upper bound of the distribution (not included)
         * @param input a reference to the input tensor that will be filled with random numbers in the range [a, b) from uniform distribution.
         */
        void uniform(const f32_t& a, const f32_t& b, const Tensor& input);

        /**
         * Generate normally distributed random values within [0, 1] both inclusive for tensor.
         * @param dim the dimension of the tensor
         * @return A tensor of type T (fp32 in default) filled with random numbers in the range [0, 1] from normal distribution
         */
        Tensor normal(const std::vector<ui32_t>& dim);

        /**
         * Generate values for a tensor from a Xavier normal distribution.
         * @note Xavier normal initialization (also known as Glorot normal initialization) is a weight
         * initialization technique that draws values from a normal distribution with a mean of 0
         * and a standard deviation of `sqrt(2 / (fan_in + fan_out))`.
         * @param dim the dimension of the tensor
         * @return A tensor of type T (fp32 in default) filled with random numbers from Xavier normal distribution
         */
        Tensor xavier_normal(const std::vector<ui32_t>& dim);

        /**
         * Generate values for a tensor from a Kaiming normal distribution.
         * @note The Kaiming normal initialization is commonly used is commonly used for layers with ReLU
         * activation function in deep neural networks.
         * @param dim the dimension of the tensor
         * @param a A floating-point value that controls the variance of the initialization (0.0 in default)
         * @return A tensor of type T (fp32 in default) filled with random numbers from Kaiming normal distribution
         */
        Tensor kaiming_normal(const std::vector<ui32_t>& dim, f32_t a = 0.0);

        /**
         * Generate values for a tensor from a Kaiming uniform distribution
         * @param dim the dimension of the tensor
         * @param a A floating-point value that controls the variance of the initialization (0.0 in default)
         * @return A tensor of type T (fp32 in default) filled with random numbers from Kaiming uniform distribution
         */
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







}


#endif
