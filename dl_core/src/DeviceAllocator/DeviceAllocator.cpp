#include "DeviceAllocator/DeviceAllocator.h"
#include <cassert>
#include <cstdlib>

namespace dl_core {
    std::shared_ptr<CpuAllocator> DeviceAllocatorFactory::create_cpu_allocator() {
        if (m_cpuAllocator == nullptr) {
            m_cpuAllocator = std::make_shared<CpuAllocator>(DeviceType::cpu);
        }

        return m_cpuAllocator;
    }

    std::shared_ptr<CudaAllocator> DeviceAllocatorFactory::create_cuda_allocator() {
        if (m_cudaAllocator == nullptr) {
            m_cudaAllocator = std::make_shared<CudaAllocator>(DeviceType::cuda);
        }

        return m_cudaAllocator;
    }

    DeviceType DeviceAllocator::device_type() const {
        return m_deviceType;
    }

    void* CpuAllocator::allocate(size_t byteSize) const {
        if (!byteSize)
            return nullptr;

        return std::malloc(byteSize);
    }

    void CpuAllocator::deallocate(void* ptr) const {
        assert(ptr != nullptr);
        std::free(ptr);
    }

    void* CudaAllocator::allocate(size_t byteSize) const {
        return nullptr;
    }

    void CudaAllocator::deallocate(void* ptr) const {

    }

    std::shared_ptr<CudaAllocator> DeviceAllocatorFactory::m_cudaAllocator = nullptr;
    std::shared_ptr<CpuAllocator> DeviceAllocatorFactory::m_cpuAllocator = nullptr;

}


