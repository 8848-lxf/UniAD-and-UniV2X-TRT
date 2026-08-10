#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env.sh"
cd "${REPRO_ROOT}/UniAD_deploy"

export PYTHONPATH="${REPRO_ROOT}/UniAD_deploy:${PYTHONPATH:-}"

exec python tools/process_metadata.py \
  --num_frame 6018 \
  --dump_folder "${REPRO_ROOT}/UniAD_deploy/nuscenes_np" \
  --dump_trt_path "${REPRO_ROOT}/UniAD_deploy/nuscenes_np/uniad_trt_input" \
  --dump_onnx_path "${REPRO_ROOT}/UniAD_deploy/nuscenes_np/uniad_onnx_input" \
  --dump_gt_path "${REPRO_ROOT}/UniAD_deploy/nuscenes_np/planning_ground_truth"
