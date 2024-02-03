# USAGE
# cmake -B build . -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

project(simulator LANGUAGES CUDA C CXX)

add_executable(cudatest 1.cu)
include_directories(cudatest /usr/local/cuda-12.1/include/)
target_compile_features(cudatest PUBLIC cxx_std_11)
set_target_properties(cudatest PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES 89) 
