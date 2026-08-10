#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env.sh"

BASE_ROOT=${BASE_ROOT:-${REPRO_ROOT}/artifacts/uniad_base_e2e}
CHECKPOINT=${1:-/home/lixingfeng/data/ckpts/uniad_base_e2e.pth}
UNIAD_GPU=${UNIAD_GPU:-6}

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing UniAD-base checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ ! -f "${BASE_ROOT}/metadata/onnx_inputs/img/0.npy" ]]; then
  echo "Missing UniAD-base export metadata under ${BASE_ROOT}/metadata" >&2
  exit 2
fi

mkdir -p "${BASE_ROOT}/onnx/pytorch_temporal_outputs"
mkdir -p "${BASE_ROOT}/onnx/raw_inputs"
mkdir -p "${BASE_ROOT}/logs"
cd "${REPRO_ROOT}/UniAD_deploy"

export CUDA_VISIBLE_DEVICES="${UNIAD_GPU}"
export PYTHONPATH="${REPRO_ROOT}/UniAD_deploy:${PYTHONPATH:-}"

exec python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=${MASTER_PORT:-29613} \
  tools/export_onnx.py \
  projects/configs/stage2_e2e/base_e2e_trt_p.py \
  "${CHECKPOINT}" \
  --launcher pytorch \
  --eval bbox \
  --onnx-output "${BASE_ROOT}/onnx/uniad_base_e2e_dcn_plugin.onnx" \
  --onnx-input-dir "${BASE_ROOT}/metadata/onnx_inputs" \
  --onnx-runtime-output-dir "${BASE_ROOT}/onnx/pytorch_temporal_outputs" \
  --raw-input-dump-dir "${BASE_ROOT}/onnx/raw_inputs"
