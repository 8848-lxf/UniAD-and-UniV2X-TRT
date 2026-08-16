#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env.sh"
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${REPRO_ROOT}/artifacts}
UNIAD_GPU=${UNIAD_GPU:-0}

CHECKPOINT=${1:-${REPRO_ROOT}/artifacts/checkpoints/tiny_imgx0.25_e2e_ep20.pth}
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing stage-2 checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${ARTIFACT_ROOT}/onnx/pytorch_temporal_outputs"
mkdir -p "${ARTIFACT_ROOT}/onnx/raw_inputs"
cd "${REPRO_ROOT}/UniAD_deploy"

export CUDA_VISIBLE_DEVICES="${UNIAD_GPU}"
export PYTHONPATH="${REPRO_ROOT}/UniAD_deploy:${PYTHONPATH:-}"

exec python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=${MASTER_PORT:-29603} \
  tools/export_onnx.py \
  projects/configs/stage2_e2e/tiny_imgx0.25_e2e_trt_p.py \
  "${CHECKPOINT}" \
  --launcher pytorch \
  --eval bbox \
  --onnx-output "${ARTIFACT_ROOT}/onnx/uniad_tiny_imgx0.25_cp.onnx" \
  --onnx-input-dir "${REPRO_ROOT}/UniAD_deploy/nuscenes_np/uniad_onnx_input" \
  --onnx-runtime-output-dir "${ARTIFACT_ROOT}/onnx/pytorch_temporal_outputs" \
  --raw-input-dump-dir "${ARTIFACT_ROOT}/onnx/raw_inputs"
