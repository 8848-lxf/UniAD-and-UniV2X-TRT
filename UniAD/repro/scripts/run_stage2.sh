#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env.sh"
cd "${REPRO_ROOT}/UniAD_train"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}
NPROC_PER_NODE=${NPROC_PER_NODE:-6}
export PYTHONPATH="${REPRO_ROOT}/UniAD_train:${PYTHONPATH:-}"

exec python -m torch.distributed.launch \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_addr=127.0.0.1 \
  --master_port=${MASTER_PORT:-29602} \
  tools/train.py \
  projects/configs/stage2_e2e/tiny_imgx0.25_e2e.py \
  --launcher pytorch \
  --deterministic \
  --work-dir ../artifacts/stage2 \
  --cfg-options load_from=../artifacts/stage1/epoch_6.pth
