#!/usr/bin/env bash

set -eo pipefail
set +u
source /home/lixingfeng/anaconda3/etc/profile.d/conda.sh
conda activate modelopt_uniad_dl4agx
set -u

export TRT_ROOT=/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118
export CUDA_HOME="${CONDA_PREFIX}"
export CC="${CONDA_PREFIX}/bin/gcc"
export CXX="${CONDA_PREFIX}/bin/g++"
export CUDAHOSTCXX="${CONDA_PREFIX}/bin/g++"
export CUDNN_ROOT="${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cudnn"
export PATH="${CUDA_HOME}/bin:${TRT_ROOT}/bin:${PATH}"
export LD_LIBRARY_PATH="${TRT_ROOT}/lib:${CUDNN_ROOT}/lib:${CUDA_HOME}/lib:${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
