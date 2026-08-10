#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env.sh"

BASE_ROOT=${BASE_ROOT:-${REPRO_ROOT}/artifacts/uniad_base_e2e}
CHECKPOINT=${1:-/home/lixingfeng/data/ckpts/uniad_base_e2e.pth}
MAX_CALIBRATION_SAMPLES=${2:-8}
UNIAD_GPU=${UNIAD_GPU:-6}

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing UniAD-base checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${BASE_ROOT}/calibration"
cd "${REPRO_ROOT}/UniAD_deploy"

export CUDA_VISIBLE_DEVICES="${UNIAD_GPU}"
export PYTHONPATH="${REPRO_ROOT}/UniAD_deploy:${PYTHONPATH:-}"

exec python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=${MASTER_PORT:-29614} \
  tools/prepare_calib_data.py \
  projects/configs/stage2_e2e/base_e2e_trt_p.py \
  "${CHECKPOINT}" \
  --launcher pytorch \
  --eval bbox \
  --dataset-split train \
  --bev-height 200 \
  --img-height 928 \
  --img-width 1600 \
  --calibration-output "${BASE_ROOT}/calibration/train_calib_shape0_901.npz" \
  --max-calibration-samples "${MAX_CALIBRATION_SAMPLES}"
