#!/usr/bin/env bash

set -eo pipefail
set +u
source /home/lixingfeng/anaconda3/etc/profile.d/conda.sh
conda activate /home/lixingfeng/UniAD_examine/.conda/univ2x_trt107_runtime
set -u

EXPECTED_PREFIX=/home/lixingfeng/UniAD_examine/.conda/univ2x_trt107_runtime
if [[ "${CONDA_PREFIX}" != "${EXPECTED_PREFIX}" ]]; then
  echo "Unexpected TensorRT 10.7 environment: ${CONDA_PREFIX}" >&2
  return 2 2>/dev/null || exit 2
fi

TRT_PYTHON_LIB="${CONDA_PREFIX}/lib/python3.8/site-packages/tensorrt_libs"
for required_path in \
  "${CONDA_PREFIX}/bin/nvcc" \
  "${CONDA_PREFIX}/bin/g++" \
  "${TRT_PYTHON_LIB}/libnvinfer.so.10"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing isolated TensorRT 10.7 toolchain input: ${required_path}" >&2
    return 2 2>/dev/null || exit 2
  fi
done

export CUDA_HOME="${CONDA_PREFIX}"
export CC="${CONDA_PREFIX}/bin/gcc"
export CXX="${CONDA_PREFIX}/bin/g++"
export CUDAHOSTCXX="${CONDA_PREFIX}/bin/g++"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${TRT_PYTHON_LIB}:${CONDA_PREFIX}/lib/python3.8/site-packages/torch/lib:${CONDA_PREFIX}/lib:${CONDA_PREFIX}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
