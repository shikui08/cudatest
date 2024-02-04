# USAGE
# cmake -B build . -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake_minimum_required(VERSION 3.22)
project(simulator LANGUAGES CUDA C CXX)

add_executable(cudatest 1.cu)
include_directories(cudatest /usr/local/cuda-12.1/include/)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++20")
target_compile_options(cudatest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       -std=c++20
                       --expt-relaxed-constexpr
                       >)
set_target_properties(cudatest PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 20
    CUDA_ARCHITECTURES 89) 
