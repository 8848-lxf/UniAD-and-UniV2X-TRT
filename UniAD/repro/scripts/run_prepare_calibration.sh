#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env.sh"
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${REPRO_ROOT}/artifacts}
UNIAD_GPU=${UNIAD_GPU:-0}

CHECKPOINT=${1:-${REPRO_ROOT}/artifacts/checkpoints/tiny_imgx0.25_e2e_ep20.pth}
MAX_CALIBRATION_SAMPLES=${2:-0}
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing stage-2 checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

mkdir -p "${ARTIFACT_ROOT}/calibration"
cd "${REPRO_ROOT}/UniAD_deploy"

export CUDA_VISIBLE_DEVICES="${UNIAD_GPU}"
export PYTHONPATH="${REPRO_ROOT}/UniAD_deploy:${PYTHONPATH:-}"

exec python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=${MASTER_PORT:-29604} \
  tools/prepare_calib_data.py \
  projects/configs/stage2_e2e/tiny_imgx0.25_e2e_trt_p.py \
  "${CHECKPOINT}" \
  --launcher pytorch \
  --eval bbox \
  --calibration-output "${ARTIFACT_ROOT}/calibration/calib_data_shape0_901.npz" \
  --max-calibration-samples "${MAX_CALIBRATION_SAMPLES}"
