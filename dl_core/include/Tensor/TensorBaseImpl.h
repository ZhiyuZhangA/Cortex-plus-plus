#ifndef TENSOR_BASE_IMPL_H
#define TENSOR_BASE_IMPL_H

#include <vector>
#include <cstdint>
#include "Buffer/Buffer.h"
#include "Dtypes/Dtype.h"

namespace cortex {
    /**
     * 可能的职责:
     * 1. Responsible for the base implementation of tensor, thus, it becomes
     * a pointer, while the layers don't connect tensor, but connect tensorBaseImpl
     * 那么梯度也需要使用impl来实现
     * Or 2. Only responsible for reduction of copy while copy tensors
     */
    class TensorBaseImpl {
    public:

    private:
        std::vector<uint32_t> m_shape;
        std::vector<uint32_t> m_stride;

        dtype m_dtype = dtype::None;
        std::shared_ptr<Buffer> m_buffer;
        uint32_t m_size = 0;

    };

}
#endif
