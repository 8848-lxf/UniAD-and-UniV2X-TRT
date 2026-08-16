#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env_modelopt.sh"

PLUGIN_ROOT="${REPRO_ROOT}/UniAD_deploy/plugins"

cmake -S "${PLUGIN_ROOT}" -B "${PLUGIN_ROOT}/build_repro" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER="${CC}" \
  -DCMAKE_CXX_COMPILER="${CXX}" \
  -DCMAKE_CUDA_COMPILER="${CUDA_HOME}/bin/nvcc" \
  -DCUDAToolkit_ROOT="${CUDA_HOME}" \
  -DCMAKE_TENSORRT_PATH="${TRT_ROOT}" \
  -DCUDNN_ROOT="${CUDNN_ROOT}" \
  -DTARGET_GPU_SM=89

cmake --build "${PLUGIN_ROOT}/build_repro" --parallel 8
cmake --install "${PLUGIN_ROOT}/build_repro"

