#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env.sh"

export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.6+PTX"
export MAX_JOBS="${MAX_JOBS:-8}"

cd "${REPRO_ROOT}/UniAD_deploy/third_party/uniad_mmdet3d"
python setup.py build_ext --inplace --force
