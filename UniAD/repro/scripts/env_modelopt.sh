#!/usr/bin/env bash

set -eo pipefail
set +u
source /home/lixingfeng/anaconda3/etc/profile.d/conda.sh
conda activate modelopt_uniad_dl4agx
set -u

if [[ -z "${TRT_ROOT:-}" ]]; then
  for candidate in \
    /home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118 \
    /home/lixingfeng/uniad-trt/py310/TensorRT-10.9_x86_cu118; do
    if [[ -f "${candidate}/targets/x86_64-linux-gnu/lib/libnvinfer.so.10" ]]; then
      TRT_ROOT="${candidate}"
      break
    fi
  done
fi
if [[ ! -f "${TRT_ROOT:-}/targets/x86_64-linux-gnu/lib/libnvinfer.so.10" ]]; then
  echo "TensorRT 10.9 user-space library root not found" >&2
  return 2 2>/dev/null || exit 2
fi
export TRT_ROOT
export MODELOPT_ROOT=${MODELOPT_ROOT:-/home/lixingfeng/UniAD_examine/UniV2X/Model-Optimizer-0.29.0}
export CUDA_HOME="${CONDA_PREFIX}"
export CC="${CONDA_PREFIX}/bin/gcc"
export CXX="${CONDA_PREFIX}/bin/g++"
export CUDAHOSTCXX="${CONDA_PREFIX}/bin/g++"
export CUDNN_ROOT="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn"
export PATH="${CUDA_HOME}/bin:${TRT_ROOT}/bin:${TRT_ROOT}/targets/x86_64-linux-gnu/bin:${PATH}"
export LD_LIBRARY_PATH="${TRT_ROOT}/lib:${TRT_ROOT}/targets/x86_64-linux-gnu/lib:${CUDNN_ROOT}/lib:${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${MODELOPT_ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
