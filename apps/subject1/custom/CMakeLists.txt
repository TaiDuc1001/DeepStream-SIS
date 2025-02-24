cmake_minimum_required(VERSION 3.10)
project(TensorRTDetection)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# TensorRT includes
# Update TENSORRT_DIR to the actual path where TensorRT is installed
set(TENSORRT_DIR /usr/lib/x86_64-linux-gnu) # Example path; modify as needed
include_directories(${TENSORRT_DIR}/include)

# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Link directories
link_directories(${TENSORRT_DIR}/lib)

add_executable(tensorrt_detection tensorrt_detection.cpp)

# Link libraries
target_link_libraries(tensorrt_detection
    ${OpenCV_LIBS}
    nvinfer
    nvparsers
    nvonnxparser
    cudart
    ${CUDA_LIBRARIES}
)