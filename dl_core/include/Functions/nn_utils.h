#ifndef NN_UTILS_H
#define NN_UTILS_H
#include "Tensor/Tensor.h"

namespace cortex {

    /**
     * All function in this file would be labeled with capitalized F,
     * indicating that they are the function that execute the actual computation for each nn layer
     */

    /**
     * This function applies linear function with bias term to the input tensor.
     * It performs a matrix multiplication between the input tensor and the weight tensor, and addition with bias term,
     * which finally produce the output tensor.
     * @param input the input tensor
     *              - Shape: \f$ (...N, D_{in}) \f$
     * @param weight the weight of the linear function
     *              - Shape: \f$ (D_{out}, D_{in}) \f$
     * @param bias the bias of the linear function
     *              - Shape: \f$ (1, D_{out}) \f$
     * @return A Tensor object representing the result of the linear transformation with bias term.
     * @throws std::invalid_argument If the dimensions of `input` and `weight` are incompatible
     *         for matrix multiplication.
     */
    Tensor FLinear(const Tensor& input, const Tensor& weight, const Tensor& bias);

    /**
     * This function applies linear transformation without bias term to the input tensor.
     * It performs a matrix multiplication between the input tensor and the weight tensor, producing the output tensor.
     * @param input the input tensor
     *              - Shape: \f$ (...N, D_{in}) \f$
     * @param weight the weight of the linear function
     *              - Shape: \f$ (D_{out}, D_{in}) \f$
     * @return A Tensor object representing the result of the linear transformation without bias term.
     * @throws std::invalid_argument If the dimensions of `input` and `weight` are incompatible
     *         for matrix multiplication.
     */
    Tensor FLinear(const Tensor& input, const Tensor& weight);

    /**
     * This function applies ReLu activation function to the input tensor.
     * @param input the input tensor
     * @return A new Tensor object with the same shape as the input tensor, where each
     *         element has been transformed using the ReLU activation function.
     */
    Tensor FReLu(const Tensor& input);

    /**
     * This function applies Leaky ReLu activation function to the input tensor.
     * @param input the input tensor
     * @param ng_slope the negative slope of leaky relu
     * @return A new Tensor object with the same shape as the input tensor, where each
     *         element has been transformed using the Leaky ReLU activation function.
     */
    Tensor FLeakyReLu(const Tensor& input, const float& ng_slope);


}

#endif //NN_UTILS_H
