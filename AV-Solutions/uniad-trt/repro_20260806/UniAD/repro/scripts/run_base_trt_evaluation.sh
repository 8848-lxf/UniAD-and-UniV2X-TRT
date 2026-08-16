#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 fp32|fp16|int8_eq_fp16 [NUM_FRAMES]" >&2
  exit 2
fi

PRECISION=$1
NUM_FRAMES=${2:-6018}
REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DEPLOY_ROOT=${UNIAD_DEPLOY_ROOT:-${REPRO_ROOT}/../../UniAD_deploy}
source "${REPRO_ROOT}/scripts/env_modelopt.sh"

BASE_ROOT=${BASE_ROOT:-${REPRO_ROOT}/artifacts/uniad_base_e2e}
UNIAD_GPU=${UNIAD_GPU:-6}
ENGINE_PATH=${ENGINE_PATH:-${BASE_ROOT}/engines/uniad_base_${PRECISION}.engine}
EVAL_TAG=${EVAL_TAG:-${PRECISION}}
# Keep host padding/profile capacity aligned with the base engine (max=1400).
FIXED_TRACK_COUNT=${FIXED_TRACK_COUNT:-1400}
TEMPORAL_PROTOCOL=${TEMPORAL_PROTOCOL:-official_literal}
COLLISION_OPTIMIZATION=${COLLISION_OPTIMIZATION:-1}
DUMP_OCCUPANCY=${DUMP_OCCUPANCY:-1}
APP_ROOT=${UNIAD_APP_ROOT:-${REPRO_ROOT}/../runtime/inference_app/enqueueV3}
APP_BUILD_DIR=${UNIAD_APP_BUILD_DIR:-build_base_recheck}
APP_PATH="${APP_ROOT}/${APP_BUILD_DIR}/uniad"
PLUGIN_PATH="${APP_ROOT}/${APP_BUILD_DIR}/libuniad_plugin.so"
INPUT_PATH="${BASE_ROOT}/metadata/trt_inputs"
OUTPUT_PATH="${BASE_ROOT}/evaluation/tensorrt_${EVAL_TAG}"
METRICS_PATH="${OUTPUT_PATH}/latency_metrics.json"
RUNTIME_WORKDIR=${UNIAD_RUNTIME_WORKDIR:-${DEPLOY_ROOT}}

for required_path in "${ENGINE_PATH}" "${APP_PATH}" "${PLUGIN_PATH}" "${INPUT_PATH}/info.txt" "${RUNTIME_WORKDIR}/data"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing UniAD-base evaluation input: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_PATH}"
export CUDA_VISIBLE_DEVICES="${UNIAD_GPU}"
export UNIAD_COLLISION_OPTIMIZATION="${COLLISION_OPTIMIZATION}"
export UNIAD_DUMP_OCCUPANCY="${DUMP_OCCUPANCY}"
cd "${RUNTIME_WORKDIR}"
"${APP_PATH}" \
  "${ENGINE_PATH}" \
  "${PLUGIN_PATH}" \
  "${INPUT_PATH}" \
  "${OUTPUT_PATH}" \
  "${NUM_FRAMES}" \
  "${METRICS_PATH}" \
  10 \
  0 \
  "${FIXED_TRACK_COUNT}" \
  "${TEMPORAL_PROTOCOL}"

source "${REPRO_ROOT}/scripts/env.sh"
REFERENCE_PATH="${BASE_ROOT}/evaluation/pytorch_fp32/planning_predictions.csv"
REFERENCE_ARGS=()
if [[ -f "${REFERENCE_PATH}" ]]; then
  REFERENCE_ARGS=(--reference-predictions "${REFERENCE_PATH}")
fi
cd "${DEPLOY_ROOT}"
python tools/evaluate_planning_outputs.py \
  --predictions "${OUTPUT_PATH}/planning_predictions.csv" \
  --ground-truth "${BASE_ROOT}/metadata/planning_ground_truth" \
  --output "${OUTPUT_PATH}/planning_metrics.json" \
  --x-bound -50 50 0.5 \
  --y-bound -50 50 0.5 \
  "${REFERENCE_ARGS[@]}"
