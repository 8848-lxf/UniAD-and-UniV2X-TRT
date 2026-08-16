#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env.sh"
cd "${REPRO_ROOT}/UniAD_deploy"

export PYTHONPATH="${REPRO_ROOT}/UniAD_deploy:${PYTHONPATH:-}"
BASE_ROOT=${BASE_ROOT:-${REPRO_ROOT}/artifacts/uniad_base_e2e}

exec python tools/process_metadata.py \
  --config projects/configs/stage2_e2e/base_e2e_trt_p.py \
  --num_frame 6018 \
  --dump_folder "${BASE_ROOT}/metadata" \
  --dump_trt_path "${BASE_ROOT}/metadata/trt_inputs" \
  --dump_onnx_path "${BASE_ROOT}/metadata/onnx_inputs" \
  --dump_gt_path "${BASE_ROOT}/metadata/planning_ground_truth"
