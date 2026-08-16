#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env.sh"

BASE_ROOT=${BASE_ROOT:-${REPRO_ROOT}/artifacts/uniad_base_e2e}
UNIAD_GPU=${UNIAD_GPU:-6}
CHECKPOINT=${1:-/home/lixingfeng/data/ckpts/uniad_base_e2e.pth}
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Missing UniAD-base checkpoint: ${CHECKPOINT}" >&2
  exit 2
fi

EVAL_DIR="${BASE_ROOT}/evaluation/pytorch_fp32"
mkdir -p "${EVAL_DIR}/show"
cd "${REPRO_ROOT}/UniAD_train"

export CUDA_VISIBLE_DEVICES="${UNIAD_GPU}"
export PYTHONPATH="${REPRO_ROOT}/UniAD_train:${PYTHONPATH:-}"
export UNIAD_BENCHMARK_DIR="${EVAL_DIR}"
export UNIAD_BENCHMARK_WARMUP=10

exec python -m torch.distributed.launch \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=${MASTER_PORT:-29616} \
  tools/test.py \
  projects/configs/stage2_e2e/base_e2e.py \
  "${CHECKPOINT}" \
  --launcher pytorch \
  --deterministic \
  --eval bbox \
  --out "${EVAL_DIR}/results.pkl" \
  --show-dir "${EVAL_DIR}/show"
