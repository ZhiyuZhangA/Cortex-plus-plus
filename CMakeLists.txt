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
        dl_core/include/Layers/Operators/AddLayer.h
        dl_core/include/Layers/Kernels/DeviceKernel.h
        dl_core/include/Random/RandomEngine.h
        dl_core/include/Layers/Kernels/cuda/add_kernel_cuda.h
        dl_core/include/Layers/Kernels/cuda/sub_kernel_cuda.h
        dl_core/include/Layers/Operators/SubLayer.h
        dl_core/include/DLEngine/DLEngine.h
        dl_core/include/Layers/Kernels/cuda/mul_kernel_cuda.h
        dl_core/include/Layers/Kernels/cuda/div_kernel_cuda.h
        dl_core/include/Layers/Operators/DivLayer.h
        dl_core/include/Layers/TransposeLayer.h
        dl_core/include/Layers/Kernels/cuda/tranpose_kernel_cuda.h
        dl_core/include/Layers/Kernels/cuda/pow_kernel_cuda.h
        dl_core/include/Layers/Functions/PowLayer.h
        dl_core/include/Functions/math_utils.h
        dl_core/include/Layers/Kernels/x86/math_kernel_cpu.h
        dl_core/include/Layers/Functions/LogLayer.h
        dl_core/include/Layers/Functions/SinLayer.h
        dl_core/include/Layers/Functions/CosLayer.h
        dl_core/include/Layers/Functions/ExpLayer.h
        dl_core/include/Layers/Operators/MulLayer.h
        dl_core/include/Layers/Functions/TanLayer.h
        dl_core/include/Layers/Functions/aTanLayer.h
        dl_core/include/Layers/BroadcastLayer.h
        dl_core/include/Layers/SumToLayer.h
        dl_core/include/Tensor/TensorBaseImpl.h
        dl_core/include/Layers/nn/LinearLayer.h
        dl_core/include/Layers/Operators/MatmulLayer.h
        dl_core/include/Functions/nn_utils.h
        dl_core/include/Layers/Kernels/x86/nn_kernel_cpu.h
        dl_core/include/Modules/Linear.h
        dl_core/include/Modules/BaseModule.h
        dl_core/include/Layers/nn/ReLuLayer.h
        dl_core/include/Modules/ReLu.h
        dl_core/include/Layers/nn/SigmoidLayer.h
        dl_core/include/Layers/nn/MSELossLayer.h
        dl_core/include/Modules/Loss/MSELoss.h
        dl_core/include/Modules/Loss/BaseLoss.h
        dl_core/include/Functions/loss.h
        dl_core/include/avx2_extension/avx2_math_ext.h
        dl_core/include/Optimizers/BaseOptimizer.h
        dl_core/include/Models/Sequential.h
        dl_core/include/Modules/Sigmoid.h
        dl_core/include/avx2_extension/avx2_common_ext.h
        dl_core/include/Layers/nn/SoftmaxLayer.h
        dl_core/include/Modules/Softmax.h
)

set(SOURCE dl_core/src/DeviceAllocator/DeviceAllocator.cpp main.cpp
        dl_core/src/Tensor/Buffer/Buffer.cpp
        dl_core/src/Tensor/Tensor.cpp
        dl_core/src/Layers/Operators/AddLayer.cpp
        dl_core/src/Layers/BaseLayer.cpp
        dl_core/src/Layers/Kernels/DeviceKernel.cpp
        dl_core/src/Layers/Kernels/x86/add_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/cuda/add_kernel_cuda.cpp
        dl_core/src/Random/RandomEngine.cpp
        dl_core/src/DLEngine/DLEngine.cpp
        dl_core/src/Layers/Kernels/x86/sub_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/cuda/sub_kernel_cuda.cpp
        dl_core/src/Layers/Operators/SubLayer.cpp
        dl_core/src/Layers/Kernels/x86/mul_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/x86/div_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/cuda/mul_kernel_cuda.cpp
        dl_core/src/Layers/Kernels/cuda/div_kernel_cuda.cpp

        dl_core/src/Layers/Operators/MulLayer.cpp
        dl_core/src/Layers/TransposeLayer.cpp
        dl_core/src/Layers/Kernels/x86/transpose_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/cuda/transpose_kernel_cuda.cpp
        dl_core/src/Layers/Operators/DivLayer.cpp
        dl_core/src/Layers/Kernels/x86/pow_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/cuda/pow_kernel_cuda.cpp
        dl_core/src/Layers/Functions/PowLayer.cpp
        dl_core/src/Functions/math_utils.cpp
        dl_core/src/Layers/Kernels/x86/exp_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/x86/log_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/x86/sin_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/x86/cos_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/x86/tan_kernel_cpu.cpp
        dl_core/src/Layers/Kernels/x86/arctan_kernel_cpu.cpp
        dl_core/src/Layers/Functions/ExpLayer.cpp
        dl_core/src/Layers/Functions/LogLayer.cpp
        dl_core/src/Layers/Functions/SinLayer.cpp
        dl_core/src/Layers/Functions/CosLayer.cpp
        dl_core/src/Layers/Functions/TanLayer.cpp
        dl_core/src/Layers/Functions/aTanLayer.cpp
        dl_core/src/Layers/BroadcastLayer.cpp
        dl_core/src/Layers/SumToLayer.cpp
        dl_core/src/Layers/Kernels/x86/sum_kernel_cpu.cpp
        dl_core/src/Graphs/autograd_graph.cpp
        dl_core/include/Graphs/autograd_graph.h
        dl_core/src/Layers/nn/LinearLayer.cpp
        dl_core/src/Layers/Kernels/x86/matmul_kernel_cpu.cpp
        dl_core/src/Layers/Operators/MatmulLayer.cpp
        dl_core/src/Functions/nn_utils.cpp
        dl_core/src/Layers/Kernels/x86/linear_kernel_cpu.cpp
        dl_core/src/Modules/Linear.cpp
        dl_core/src/Layers/nn/ReLuLayer.cpp
        dl_core/src/Layers/Kernels/x86/relu_kernel_cpu.cpp
        dl_core/src/Modules/ReLu.cpp
        dl_core/src/Layers/nn/SigmoidLayer.cpp
        dl_core/src/Functions/loss.cpp
        dl_core/src/Layers/nn/MSELossLayer.cpp
        dl_core/src/Layers/Kernels/x86/mse_loss_kernel_cpu.cpp
        dl_core/src/Modules/Loss/MSELoss.cpp
        Examples/BostonHousingPredictions.cpp
        dl_core/include/Models/BaseModel.h
        dl_core/src/Layers/Kernels/x86/sigmoid_kernel_cpu.cpp
        dl_core/src/Modules/Sigmoid.cpp
        dl_core/src/Layers/Kernels/x86/softmax_kernel_cpu.cpp
        dl_core/src/Layers/nn/SoftmaxLayer.cpp
        dl_core/src/Modules/Softmax.cpp
)

link_directories(${PROJECT_SOURCE_DIR}/thirdparty/OpenBLAS-0.3.28-x64-64/lib)

add_executable(Cortex++ ${SOURCE} ${HEADERS})

target_link_libraries(Cortex++ openblas)
