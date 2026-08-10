#!/usr/bin/env bash

set -eo pipefail
set +u

source /home/lixingfeng/anaconda3/etc/profile.d/conda.sh
conda activate torch112
set -u

export CUDA_HOME="${CONDA_PREFIX}"
export CC="${CONDA_PREFIX}/bin/gcc"
export CXX="${CONDA_PREFIX}/bin/g++"
export CUDAHOSTCXX="${CONDA_PREFIX}/bin/g++"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
export PYTHONHASHSEED=0
export TZ=Asia/Shanghai
