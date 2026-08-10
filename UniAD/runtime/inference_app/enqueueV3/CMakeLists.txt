cmake_minimum_required(VERSION 3.12)
project(uniad LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type")
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
find_package(CUDAToolkit REQUIRED)

set(UNIAD_BEV_H 50 CACHE STRING "UniAD BEV height")
set(UNIAD_BEV_W 50 CACHE STRING "UniAD BEV width")
set(UNIAD_IMG_H 256 CACHE STRING "UniAD padded input image height")
set(UNIAD_IMG_W 416 CACHE STRING "UniAD padded input image width")
set(UNIAD_IMAGE_RESIZE_SCALE 0.25 CACHE STRING "Raw nuScenes image resize scale")
add_compile_definitions(
  UNIAD_BEV_H=${UNIAD_BEV_H}
  UNIAD_BEV_W=${UNIAD_BEV_W}
  UNIAD_IMG_H=${UNIAD_IMG_H}
  UNIAD_IMG_W=${UNIAD_IMG_W}
  UNIAD_IMAGE_RESIZE_SCALE=${UNIAD_IMAGE_RESIZE_SCALE}
)

if(NOT TENSORRT_PATH)
  message(FATAL_ERROR "Need: -DTENSORRT_PATH=/path/to/TensorRT")
endif()

if(NOT TARGET_GPU_SM)
  message(FATAL_ERROR "Need: -DTARGET_GPU_SM=GPU_compute_capability")
endif()

set(CUDNN_ROOT "" CACHE PATH "Directory containing cuDNN lib and include trees")
set(TENSORRT_INCLUDE_DIRS ${TENSORRT_PATH}/include/)
set(TENSORRT_LIBRARY_DIRS ${TENSORRT_PATH}/lib/)
set(STB_LIB_PTH ../../../common/dependencies/stb/)
set(CUOSD_LIB_PTH ../../../common/dependencies/cuOSD/)
find_file(CUDNN_LIBRARY
  NAMES libcudnn.so.9 libcudnn.so
  PATHS ${CUDNN_ROOT}/lib ${CUDNN_ROOT}
  NO_DEFAULT_PATH
  REQUIRED)

add_compile_options(-Wno-deprecated-declarations)
add_compile_definitions(ENABLE_TEXT_BACKEND_STB WIN_INTERFACE_CUSTOM)

include_directories(
  include
  src
  ${CUDAToolkit_INCLUDE_DIRS}
  ${CUDNN_ROOT}/include
  ${STB_LIB_PTH}
  ${CUOSD_LIB_PTH}/src
  ${TENSORRT_INCLUDE_DIRS}
  ../../dependencies/BEVFormer_tensorrt/TensorRT/common
)
link_directories(${TENSORRT_LIBRARY_DIRS})

file(GLOB_RECURSE KERNEL_FILES 
  src/uniad.cpp
  src/tensorrt.cpp
  src/tensor.cu
  src/pre_process.cu
)

file(GLOB_RECURSE VISUALIZE_FILES 
  ${CUOSD_LIB_PTH}/src/*.cpp
  ${CUOSD_LIB_PTH}/src/*.cu
  src/visualize.cu
)

file(GLOB_RECURSE PLUGIN_FILES 
  ../../dependencies/BEVFormer_tensorrt/TensorRT/common/*.cpp 
  ../../dependencies/BEVFormer_tensorrt/TensorRT/common/*.cu 
  ../../dependencies/BEVFormer_tensorrt/TensorRT/plugin/*/*.cu 
  ../../dependencies/BEVFormer_tensorrt/TensorRT/plugin/*/*.cpp
)

set(SHARED_TARGET uniad_plugin)

add_library(${SHARED_TARGET} SHARED ${PLUGIN_FILES})
target_compile_options(${SHARED_TARGET} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${TARGET_GPU_SM}>)

target_link_libraries(${SHARED_TARGET} PUBLIC ${CUDNN_LIBRARY} nvinfer CUDA::cublas)
set_target_properties(${SHARED_TARGET} PROPERTIES CUDA_ARCHITECTURES "${TARGET_GPU_SM}")

install(
  TARGETS ${SHARED_TARGET}
  LIBRARY DESTINATION ${CMAKE_SOURCE_DIR}/build/
)

add_library(uniad_kernel SHARED
  ${KERNEL_FILES}
)

target_compile_options(uniad_kernel PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${TARGET_GPU_SM}>)

target_link_libraries(uniad_kernel
  CUDA::cudart
  CUDA::cublasLt
  nvinfer
  nvinfer_plugin
)

set_target_properties(uniad_kernel PROPERTIES CUDA_ARCHITECTURES "${TARGET_GPU_SM}")

add_executable(uniad
  src/main.cpp
  ${VISUALIZE_FILES}
)
set_target_properties(uniad PROPERTIES CUDA_ARCHITECTURES "${TARGET_GPU_SM}")

target_link_libraries(uniad
  uniad_kernel
  nvinfer
  nvinfer_plugin
)
