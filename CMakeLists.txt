cmake_minimum_required(VERSION 3.30)
project(Cortex++)

include_directories(${PROJECT_SOURCE_DIR}/dl_core/include)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/googletest/googletest/include)
include_directories(${PROJECT_SOURCE_DIR}/thirdparty/OpenBLAS-0.3.28-x64-64/include)

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/googletest)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=haswell -mfma -mavx2")

set(HEADERS
        dl_core/include/Tensor/Tensor.h
        dl_core/include/Dtypes/Dtype.h
        dl_core/include/Tensor/Buffer/Buffer.h
        dl_core/include/Common/NoCopyable.h
        dl_core/include/DeviceAllocator/DeviceAllocator.h
        dl_core/include/Tensor/Tensor_utils.h
        dl_core/include/Layers/BaseLayer.h
        dl_core/include/Layers/AddLayer.h
        dl_core/include/Layers/Kernels/DeviceKernel.h
        dl_core/include/Random/RandomEngine.h
        dl_core/include/Layers/Kernels/cuda/add_kernel_cuda.h
        dl_core/include/Layers/Kernels/cuda/sub_kernel_cuda.h
        dl_core/include/Layers/SubLayer.h
        dl_core/include/DLEngine/DLEngine.h
        dl_core/include/Layers/Kernels/cuda/mul_kernel_cuda.h
        dl_core/include/Layers/Kernels/cuda/div_kernel_cuda.h
        dl_core/include/Layers/DivLayer.h
        dl_core/include/Layers/TransposeLayer.h
        dl_core/include/Layers/Kernels/cuda/tranpose_kernel_cuda.h
        dl_core/include/Layers/Kernels/cuda/pow_kernel_cuda.h
        dl_core/include/Layers/PowLayer.h
        dl_core/include/Tensor/math_utils.h
        dl_core/include/Layers/Kernels/x86/math_kernel_cpu.h
        dl_core/include/Layers/LogLayer.h
        dl_core/include/Layers/SinLayer.h
        dl_core/include/Layers/CosLayer.h
        dl_core/include/Layers/ExpLayer.h
        dl_core/include/Layers/MulLayer.h
        dl_core/include/Layers/TanLayer.h
        dl_core/include/Layers/aTanLayer.h
        dl_core/include/Layers/BroadcastLayer.h
        dl_core/include/Layers/SumToLayer.h
)

set(SOURCE dl_core/src/DeviceAllocator/DeviceAllocator.cpp main.cpp
        dl_core/src/Tensor/Buffer/Buffer.cpp
        dl_core/src/Tensor/Tensor.cpp
        dl_core/src/Layers/AddLayer.cpp
        dl_core/src/Layers/BaseLayer.cpp
        dl_core/src/Layers/Kernels/DeviceKernel.cpp
        dl_core/src/Layers/Kernels/x86/add_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/cuda/add_kernel_cuda.cpp
        dl_core/src/Random/RandomEngine.cpp
        dl_core/src/DLEngine/DLEngine.cpp
        dl_core/src/Layers/Kernels/x86/sub_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/cuda/sub_kernel_cuda.cpp
        dl_core/src/Layers/SubLayer.cpp
        dl_core/src/Layers/Kernels/x86/mul_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/x86/div_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/cuda/mul_kernel_cuda.cpp
        dl_core/src/Layers/Kernels/cuda/div_kernel_cuda.cpp

        dl_core/src/Layers/MulLayer.cpp
        dl_core/src/Layers/TransposeLayer.cpp
        dl_core/src/Layers/Kernels/x86/transpose_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/cuda/transpose_kernel_cuda.cpp
        dl_core/src/Layers/DivLayer.cpp
        dl_core/src/Layers/Kernels/x86/pow_kernel_cpu.cpp
        dl_core/include/avx2_extension/avx2_math_ext.h
        dl_core/src/Layers/Kernels/cuda/pow_kernel_cuda.cpp
        dl_core/src/Layers/PowLayer.cpp
        dl_core/src/Tensor/math_utils.cpp
        dl_core/src/Layers/Kernels/x86/exp_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/x86/log_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/x86/sin_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/x86/cos_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/x86/tan_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/x86/arctan_kernel_cpu.cpp
        dl_core/src/Layers/ExpLayer.cpp
        dl_core/src/Layers/LogLayer.cpp
        dl_core/src/Layers/SinLayer.cpp
        dl_core/src/Layers/CosLayer.cpp
        dl_core/src/Layers/TanLayer.cpp
        dl_core/src/Layers/aTanLayer.cpp
        dl_core/src/Layers/BroadcastLayer.cpp
        dl_core/src/Layers/SumToLayer.cpp
        dl_core/src/Layers/Kernels/x86/sum_kernel_cpu.cpp
)

link_directories(${PROJECT_SOURCE_DIR}/thirdparty/OpenBLAS-0.3.28-x64-64/lib)

add_executable(Cortex++ ${SOURCE} ${HEADERS})

target_link_libraries(Cortex++ openblas)
