#ifndef NN_UTILS_H
#define NN_UTILS_H
#include "Tensor/Tensor.h"

namespace cortex {
    /**
     *
     * @param input
     * @param weight
     * @param bias
     * @return
     */
    Tensor Linear(Tensor input, Tensor weight, Tensor bias);

}

#endif //NN_UTILS_H
