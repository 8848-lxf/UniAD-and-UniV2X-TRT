#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEPLOY_ROOT=${UNIAD_DEPLOY_ROOT:-${REPRO_ROOT}/../../UniAD_deploy}
source "${REPRO_ROOT}/scripts/env.sh"

BASE_ROOT=${BASE_ROOT:-${REPRO_ROOT}/artifacts/uniad_base_e2e}
CHECKPOINT=${1:-/home/lixingfeng/data/ckpts/uniad_base_e2e.pth}
NUM_FRAMES=${2:-6018}
TEMPORAL_PROTOCOL=${TEMPORAL_PROTOCOL:-official_literal}
UNIAD_GPU=${UNIAD_GPU:-5}
OUTPUT_PATH=${OUTPUT_PATH:-${BASE_ROOT}/evaluation/pytorch_forward_${TEMPORAL_PROTOCOL}/planning_predictions_raw.csv}

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing UniAD-base checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"
export CUDA_VISIBLE_DEVICES="${UNIAD_GPU}"
export PYTHONPATH="${DEPLOY_ROOT}:${PYTHONPATH:-}"
cd "${DEPLOY_ROOT}"

exec python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=${MASTER_PORT:-29615} \
  tools/prepare_calib_data.py \
  projects/configs/stage2_e2e/base_e2e_trt_p.py \
  "${CHECKPOINT}" \
  --launcher pytorch \
  --eval bbox \
  --dataset-split test \
  --bev-height 200 \
  --img-height 928 \
  --img-width 1600 \
  --workers 8 \
  --skip-calibration-output \
  --planning-predictions-output "${OUTPUT_PATH}" \
  --max-frames "${NUM_FRAMES}" \
  --temporal-protocol "${TEMPORAL_PROTOCOL}"
