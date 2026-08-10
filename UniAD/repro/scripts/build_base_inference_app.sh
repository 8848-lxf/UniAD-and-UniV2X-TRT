#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env_modelopt.sh"

APP_ROOT="${REPRO_ROOT}/package/uniad-trt/inference_app/enqueueV3"
BUILD_ROOT="${APP_ROOT}/build_base"

cmake -S "${APP_ROOT}" -B "${BUILD_ROOT}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${CUDA_HOME}" \
  -DTENSORRT_PATH="${TRT_ROOT}" \
  -DCUDNN_ROOT="${CUDNN_ROOT}" \
  -DTARGET_GPU_SM=89 \
  -DUNIAD_BEV_H=200 \
  -DUNIAD_BEV_W=200 \
  -DUNIAD_IMG_H=928 \
  -DUNIAD_IMG_W=1600 \
  -DUNIAD_IMAGE_RESIZE_SCALE=1.0

cmake --build "${BUILD_ROOT}" --parallel 8
