#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env_trt107.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 ENGINE_PATH OUTPUT_DIRECTORY [NUM_FRAMES]" >&2
  exit 2
fi

ENGINE_PATH=$(realpath "$1")
OUTPUT_PATH=$(realpath -m "$2")
NUM_FRAMES=${3:-6018}
UNIAD_GPU=${UNIAD_GPU:-5}
FIXED_TRACK_COUNT=${FIXED_TRACK_COUNT:-1600}
RUNTIME_ROOT=${UNIAD_EXACT107_RUNTIME_ROOT:-/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/trt107/runtime_build_trt107_exact}
APP_PATH="${RUNTIME_ROOT}/uniad"
PLUGIN_PATH="${RUNTIME_ROOT}/libuniad_plugin.so"
METADATA_ROOT=${UNIAD_METADATA_ROOT:-/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/metadata_scene_reset}
INPUT_PATH="${METADATA_ROOT}/trt_inputs"
GROUND_TRUTH_PATH="${METADATA_ROOT}/planning_ground_truth"
REFERENCE_PATH=${UNIAD_REFERENCE_PREDICTIONS:-/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/evaluation/deployment_pytorch_trtp_full6018/planning_predictions.csv}
METRICS_PATH="${OUTPUT_PATH}/latency_metrics.json"

for required_path in \
  "${ENGINE_PATH}" "${APP_PATH}" "${PLUGIN_PATH}" \
  "${INPUT_PATH}/info.txt" "${REFERENCE_PATH}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing exact-10.7 evaluation input: ${required_path}" >&2
    exit 2
  fi
done

if [[ -e "${OUTPUT_PATH}" ]]; then
  echo "Refusing to overwrite an existing evaluation directory: ${OUTPUT_PATH}" >&2
  exit 2
fi
mkdir -p "${OUTPUT_PATH}"
export CUDA_VISIBLE_DEVICES="${UNIAD_GPU}"
# The original UniAD evaluation config uses use_col_optim=True. Keep the
# deployment evaluator on that protocol by default; set this to 0 explicitly
# only when collecting a raw outs_planning diagnostic.
export UNIAD_COLLISION_OPTIMIZATION="${UNIAD_COLLISION_OPTIMIZATION:-1}"

cd "${REPRO_ROOT}/UniAD_deploy"
"${APP_PATH}" \
  "${ENGINE_PATH}" \
  "${PLUGIN_PATH}" \
  "${INPUT_PATH}" \
  "${OUTPUT_PATH}" \
  "${NUM_FRAMES}" \
  "${METRICS_PATH}" \
  20 \
  0 \
  "${FIXED_TRACK_COUNT}" 2>&1 | tee "${OUTPUT_PATH}/run.log"

source "${REPRO_ROOT}/scripts/env.sh"
python tools/evaluate_planning_outputs.py \
  --predictions "${OUTPUT_PATH}/planning_predictions.csv" \
  --ground-truth "${GROUND_TRUTH_PATH}" \
  --reference-predictions "${REFERENCE_PATH}" \
  --output "${OUTPUT_PATH}/planning_metrics_vs_deployment_pytorch.json" \
  > "${OUTPUT_PATH}/planning_evaluation.log"
