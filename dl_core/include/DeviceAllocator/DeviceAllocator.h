#ifndef DEVICEALLOCATOR_H
#define DEVICEALLOCATOR_H
#include <memory>

namespace cortex_core {
    enum class DeviceType {
        cpu,
        cuda,
        Unknown,
    };

#define MEM_ALIGN 64

    class DeviceAllocator {
    public:
        explicit DeviceAllocator(const DeviceType deviceType) : m_deviceType(deviceType) {}

        virtual void* allocate(size_t byteSize) const = 0;

        virtual void deallocate(void* ptr) const = 0;

        DeviceType device_type() const;

    protected:
        DeviceType m_deviceType = DeviceType::Unknown;
    };

    class CpuAllocator : public DeviceAllocator {
    public:
        explicit CpuAllocator(const DeviceType deviceType) : DeviceAllocator(deviceType) {}

        void* allocate(size_t byteSize) const override;

        void deallocate(void* ptr) const override;
    };

    class CudaAllocator : public DeviceAllocator {
    public:
        explicit CudaAllocator(const DeviceType deviceType) : DeviceAllocator(deviceType) {}

        void* allocate(size_t byteSize) const override;

        void deallocate(void* ptr) const override;
    };

    class DeviceAllocatorFactory {
    public:
        static std::shared_ptr<CpuAllocator> create_cpu_allocator();
        static std::shared_ptr<CudaAllocator> create_cuda_allocator();

    private:
        static std::shared_ptr<CpuAllocator> m_cpuAllocator;
        static std::shared_ptr<CudaAllocator> m_cudaAllocator;
    };

}

#endif
