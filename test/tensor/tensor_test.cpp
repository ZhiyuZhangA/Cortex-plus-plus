#include "gtest/gtest.h"
#include "Tensor/Tensor.h"

namespace dl_core {
    TEST(Tensor_Test, Indices_Test) {
        Tensor tensor({2, 2, 3}, dtype::f32);
        tensor.initialize_with<f32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
        const auto val = tensor.at<f32_t>({1, 1, 2});
        EXPECT_EQ(val, 12.0f);
    }
}
