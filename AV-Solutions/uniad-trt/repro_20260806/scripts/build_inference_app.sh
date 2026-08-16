#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env_modelopt.sh"

APP_ROOT="${REPRO_ROOT}/package/uniad-trt/inference_app/enqueueV3"

cmake -S "${APP_ROOT}" -B "${APP_ROOT}/build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${CUDA_HOME}" \
  -DTENSORRT_PATH="${TRT_ROOT}" \
  -DCUDNN_ROOT="${CUDNN_ROOT}" \
  -DTARGET_GPU_SM=89

cmake --build "${APP_ROOT}/build" --parallel 8

