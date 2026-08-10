#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env.sh"
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${REPRO_ROOT}/artifacts}
UNIAD_GPU=${UNIAD_GPU:-0}
DISABLE_OCC_FOR_PLANNING_ONLY=${DISABLE_OCC_FOR_PLANNING_ONLY:-0}

CHECKPOINT=${1:-${REPRO_ROOT}/artifacts/checkpoints/tiny_imgx0.25_e2e_ep20.pth}
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing stage-2 checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

EVAL_DIR="${ARTIFACT_ROOT}/evaluation/pytorch_fp32"
mkdir -p "${EVAL_DIR}/show"
cd "${REPRO_ROOT}/UniAD_train"

export CUDA_VISIBLE_DEVICES="${UNIAD_GPU}"
export PYTHONPATH="${REPRO_ROOT}/UniAD_train:${PYTHONPATH:-}"
export UNIAD_BENCHMARK_DIR="${EVAL_DIR}"
export UNIAD_BENCHMARK_WARMUP=10

EXTRA_ARGS=()
if [[ "${DISABLE_OCC_FOR_PLANNING_ONLY}" == "1" ]]; then
  EXTRA_ARGS=(--cfg-options model.occ_head=None)
fi

exec python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=${MASTER_PORT:-29605} \
  tools/test.py \
  projects/configs/stage2_e2e/tiny_imgx0.25_e2e.py \
  "${CHECKPOINT}" \
  --launcher pytorch \
  --deterministic \
  --eval bbox \
  --out "${EVAL_DIR}/results.pkl" \
  --show-dir "${EVAL_DIR}/show" \
  "${EXTRA_ARGS[@]}"
