//
// Created by zzy on 2024/12/27.
//

#include "Tensor/Buffer/Buffer.h"

namespace cortex_core {
    Buffer::Buffer(const size_t byteSize, const std::shared_ptr<DeviceAllocator>& alloc): NonCopyable(), m_byteSize(byteSize), m_alloc(alloc) {
        m_deviceType = alloc->device_type();
        m_ptr = nullptr;
    }

    Buffer::~Buffer() {
        if (m_ptr != nullptr) {
            this->m_alloc->deallocate(m_ptr);
        }
    }

    bool Buffer::allocate() {
        this->m_ptr = this->m_alloc->allocate(this->m_byteSize);
        if (this->m_ptr == nullptr)
            return false;
        return true;
    }

    void*& Buffer::data() {
        return this->m_ptr;
    }

    size_t Buffer::byte_size() const {
        return this->m_byteSize;
    }

    DeviceType Buffer::device_type() const {
        return this->m_deviceType;
    }

    std::shared_ptr<DeviceAllocator> Buffer::device_alloc() const {
        return this->m_alloc;
    }
}
