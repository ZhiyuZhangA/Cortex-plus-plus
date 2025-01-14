#ifndef NN_UTILS_H
#define NN_UTILS_H
#include "Tensor/Tensor.h"

namespace cortex {

    /**
     * All function in this file would be labeled with capitalized F,
     * indicating that they are the function that execute the actual computation for each nn layer
     */

    /**
     * Returns a tensor after the computation of linear layer with bias
     * @param input
     * @param weight
     * @param bias
     * @return
     */
    Tensor FLinear(const Tensor& input, const Tensor& weight, const Tensor& bias);

    /**
     * Returns a tensor after the computation of linear layer without bias
     * @param input
     * @param weight
     * @return
     */
    Tensor FLinear(const Tensor& input, const Tensor& weight);


    Tensor FReLu(const Tensor& input);


}

#endif //NN_UTILS_H
