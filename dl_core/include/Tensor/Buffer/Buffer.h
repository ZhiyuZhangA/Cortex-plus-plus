#ifndef BUFFER_H
#define BUFFER_H

#include <memory>
#include "Common/NoCopyable.h"
#include "DeviceAllocator/DeviceAllocator.h"

namespace cortex_core {
    class Buffer : NonCopyable, std::enable_shared_from_this<Buffer> {
    public:
        Buffer(size_t byteSize, const std::shared_ptr<DeviceAllocator>& alloc);
        ~Buffer();
        bool allocate();
        void*& data();
        size_t byte_size() const;
        DeviceType device_type() const;
        std::shared_ptr<DeviceAllocator> device_alloc() const;

    private:
        void* m_ptr;  // Row-major order
        size_t m_byteSize;
        std::shared_ptr<DeviceAllocator> m_alloc;
        DeviceType m_deviceType = DeviceType::Unknown;
    };
}


#endif
