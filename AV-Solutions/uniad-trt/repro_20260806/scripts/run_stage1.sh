#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env.sh"
cd "${REPRO_ROOT}/UniAD_train"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
NPROC_PER_NODE=${NPROC_PER_NODE:-6}
export PYTHONPATH="${REPRO_ROOT}/UniAD_train:${PYTHONPATH:-}"

TRAIN_ARGS=(
  projects/configs/stage1_track_map/tiny_imgx0.25_track_map.py
  --launcher pytorch
  --deterministic
  --work-dir ../artifacts/stage1
)
if [[ -n "${RESUME_FROM:-}" ]]; then
  if [[ ! -s "${RESUME_FROM}" ]]; then
    echo "Missing resume checkpoint: ${RESUME_FROM}" >&2
    exit 2
  fi
  TRAIN_ARGS+=(--resume-from "${RESUME_FROM}")
fi

exec python -m torch.distributed.launch \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr=127.0.0.1 \
  --master_port=${MASTER_PORT:-29601} \
  tools/train.py \
  "${TRAIN_ARGS[@]}"
