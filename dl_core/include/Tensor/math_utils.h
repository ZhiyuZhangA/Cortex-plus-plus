#ifndef MATH_UTILS_H
#define MATH_UTILS_H
#include "Tensor.h"

namespace dl_core {

#define e 2.71828181828459045

    Tensor pow(const Tensor& base, const Tensor& n);

    Tensor pow(const Tensor& base, const f32_t& n);

    Tensor exp(const Tensor& x);

    Tensor log(const Tensor& x);

    Tensor log2(const Tensor& x);

    Tensor log10(const Tensor& x);

    Tensor sin(const Tensor& x);

    Tensor cos(const Tensor& x);

    Tensor tan(const Tensor& x);

}

#endif //MATH_UTILS_H
