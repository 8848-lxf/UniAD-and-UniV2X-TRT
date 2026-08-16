#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env.sh"

CHECKPOINT=${1:-${REPRO_ROOT}/artifacts/checkpoints/tiny_imgx0.25_e2e_ep20.pth}
AUDIT_OUTPUT=${2:-${REPRO_ROOT}/artifacts/audits/forward_uniad_trt.npz}
TEMPORAL_PROTOCOL=${TEMPORAL_PROTOCOL:-official_literal}
NUM_FRAMES=${NUM_FRAMES:-0}
UNIAD_GPU=${UNIAD_GPU:-0}

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing stage-2 checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi
if [[ -e "${AUDIT_OUTPUT}" ]]; then
  echo "Refusing to overwrite audit output: ${AUDIT_OUTPUT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${AUDIT_OUTPUT}")"
cd "${REPRO_ROOT}/UniAD_deploy"
export CUDA_VISIBLE_DEVICES="${UNIAD_GPU}"
export PYTHONPATH="${REPRO_ROOT}/UniAD_deploy:${PYTHONPATH:-}"

exec python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port="${MASTER_PORT:-29614}" \
  tools/prepare_calib_data.py \
  projects/configs/stage2_e2e/tiny_imgx0.25_e2e_trt_p.py \
  "${CHECKPOINT}" \
  --launcher pytorch \
  --dataset-split test \
  --workers-per-gpu "${UNIAD_DATALOADER_WORKERS:-8}" \
  --temporal-protocol "${TEMPORAL_PROTOCOL}" \
  --audit-only \
  --audit-output "${AUDIT_OUTPUT}" \
  --max-dataset-frames "${NUM_FRAMES}"
