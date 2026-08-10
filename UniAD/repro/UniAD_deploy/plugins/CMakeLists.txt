cmake_minimum_required(VERSION 3.12)

project(tensorrt_ops LANGUAGES CXX CUDA)
find_package(CUDAToolkit REQUIRED)

set(TARGET_GPU_SM "89" CACHE STRING "CUDA SM architecture")
set(CUDNN_ROOT "" CACHE PATH "Directory containing cuDNN lib and include trees")

if(NOT CMAKE_TENSORRT_PATH)
    message(FATAL_ERROR "Need: -DCMAKE_TENSORRT_PATH=/path/to/TensorRT")
endif()

include_directories(common)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(${CUDNN_ROOT}/include)

message(STATUS "Linking TensorRT directory: ${CMAKE_TENSORRT_PATH}")
link_directories(${CMAKE_TENSORRT_PATH}/lib)
include_directories(${CMAKE_TENSORRT_PATH}/include)

find_file(CUDNN_LIBRARY
    NAMES libcudnn.so.9 libcudnn.so
    PATHS ${CUDNN_ROOT}/lib ${CUDNN_ROOT}
    NO_DEFAULT_PATH
    REQUIRED)

file(GLOB_RECURSE TENSORRT_OPS_SRCS common/*.cpp common/*.cu plugin/*/*.cu plugin/*/*.cpp)

set(SHARED_TARGET tensorrt_ops)
add_library(${SHARED_TARGET} SHARED ${TENSORRT_OPS_SRCS})
target_compile_options(${SHARED_TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${TARGET_GPU_SM}>)

target_link_libraries(${SHARED_TARGET} PUBLIC ${CUDNN_LIBRARY} nvinfer CUDA::cublas)
set_target_properties(${SHARED_TARGET} PROPERTIES
    CUDA_ARCHITECTURES "${TARGET_GPU_SM}"
    OUTPUT_NAME "_uniad_plugins_trt10.9_x86_cu118")

install(
        TARGETS ${SHARED_TARGET}
        LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/lib
)
