#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env_modelopt.sh"

APP_ROOT=${UNIAD_APP_ROOT:-${REPRO_ROOT}/../runtime/inference_app/enqueueV3}
BUILD_ROOT="${APP_ROOT}/build_base"
AV_SOLUTIONS_ROOT=${UNIAD_AV_SOLUTIONS_ROOT:-${REPRO_ROOT}/../../../DL4AGX/AV-Solutions}
STB_ROOT=${UNIAD_STB_ROOT:-${AV_SOLUTIONS_ROOT}/common/dependencies/stb}
CUOSD_ROOT=${UNIAD_CUOSD_ROOT:-${AV_SOLUTIONS_ROOT}/far3d-trt/inference_app/dependencies/Lidar_AI_Solution/libraries/cuOSD}
BEVFORMER_TRT_ROOT=${UNIAD_BEVFORMER_TRT_ROOT:-${AV_SOLUTIONS_ROOT}/uniad-trt/dependencies/BEVFormer_tensorrt}
JPEG_ROOT=${UNIAD_JPEG_ROOT:-}
if [[ -z "${JPEG_ROOT}" ]]; then
  CONDA_ROOT=$(dirname "$(dirname "${CONDA_PREFIX}")")
  for candidate in "${CONDA_PREFIX}" "${CONDA_ROOT}"/pkgs/libjpeg-turbo-*; do
    if [[ -f "${candidate}/include/jpeglib.h" && -f "${candidate}/lib/libjpeg.so" ]]; then
      JPEG_ROOT=${candidate}
      break
    fi
  done
fi
if [[ ! -f "${APP_ROOT}/CMakeLists.txt" ]]; then
  echo "UniAD runtime source not found: ${APP_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${JPEG_ROOT:-}/include/jpeglib.h" ]]; then
  echo "Conda libjpeg-turbo development files not found; set UNIAD_JPEG_ROOT" >&2
  exit 2
fi
for dependency in "${STB_ROOT}/stb_image.h" "${CUOSD_ROOT}/src/cuosd.h" "${BEVFORMER_TRT_ROOT}/TensorRT/common/checkMacrosPlugin.cpp"; do
  if [[ ! -f "${dependency}" ]]; then
    echo "UniAD runtime dependency not found: ${dependency}" >&2
    exit 2
  fi
done

cmake -S "${APP_ROOT}" -B "${BUILD_ROOT}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${CUDA_HOME}" \
  -DTENSORRT_PATH="${TRT_ROOT}" \
  -DCUDNN_ROOT="${CUDNN_ROOT}" \
  -DUNIAD_USE_LIBJPEG_DECODER=ON \
  -DUNIAD_JPEG_ROOT="${JPEG_ROOT}" \
  -DUNIAD_STB_ROOT="${STB_ROOT}" \
  -DUNIAD_CUOSD_ROOT="${CUOSD_ROOT}" \
  -DUNIAD_BEVFORMER_TRT_ROOT="${BEVFORMER_TRT_ROOT}" \
  -DTARGET_GPU_SM=89 \
  -DUNIAD_BEV_H=200 \
  -DUNIAD_BEV_W=200 \
  -DUNIAD_IMG_H=928 \
  -DUNIAD_IMG_W=1600 \
  -DUNIAD_IMAGE_RESIZE_SCALE=1.0

cmake --build "${BUILD_ROOT}" --parallel 8
