#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 fp32|fp16|int8_eq_fp16 [NUM_FRAMES]" >&2
  exit 2
fi

PRECISION=$1
NUM_FRAMES=${2:-6018}
REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${REPRO_ROOT}/artifacts}
UNIAD_GPU=${UNIAD_GPU:-0}

case "${PRECISION}" in
  fp32)
    ENGINE_NAME=uniad_tiny_fp32.engine
    ;;
  fp16)
    ENGINE_NAME=uniad_tiny_fp16.engine
    ;;
  int8_eq_fp16)
    ENGINE_NAME=uniad_tiny_int8_eq_fp16.engine
    ;;
  *)
    echo "Unsupported precision: ${PRECISION}" >&2
    exit 2
    ;;
esac

source "${REPRO_ROOT}/scripts/env_modelopt.sh"

ENGINE_PATH="${ARTIFACT_ROOT}/engines/${ENGINE_NAME}"
APP_ROOT="${REPRO_ROOT}/package/uniad-trt/inference_app/enqueueV3"
APP_PATH="${APP_ROOT}/build/uniad"
PLUGIN_PATH="${APP_ROOT}/build/libuniad_plugin.so"
INPUT_PATH="${REPRO_ROOT}/UniAD_deploy/nuscenes_np/uniad_trt_input"
OUTPUT_PATH="${ARTIFACT_ROOT}/evaluation/tensorrt_${PRECISION}"
METRICS_PATH="${OUTPUT_PATH}/latency_metrics.json"

for required_path in "${ENGINE_PATH}" "${APP_PATH}" "${PLUGIN_PATH}" "${INPUT_PATH}/info.txt"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing TensorRT evaluation input: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_PATH}"
export CUDA_VISIBLE_DEVICES="${UNIAD_GPU}"
cd "${APP_ROOT}"
"${APP_PATH}" \
  "${ENGINE_PATH}" \
  "${PLUGIN_PATH}" \
  "${INPUT_PATH}" \
  "${OUTPUT_PATH}" \
  "${NUM_FRAMES}" \
  "${METRICS_PATH}" \
  10 \
  0

source "${REPRO_ROOT}/scripts/env.sh"
REFERENCE_ROOT=${REFERENCE_ROOT:-${ARTIFACT_ROOT}}
REFERENCE_PATH="${REFERENCE_ROOT}/evaluation/pytorch_fp32/planning_predictions.csv"
REFERENCE_ARGS=()
if [[ -f "${REFERENCE_PATH}" ]]; then
  REFERENCE_ARGS=(--reference-predictions "${REFERENCE_PATH}")
fi

cd "${REPRO_ROOT}/UniAD_deploy"
python tools/evaluate_planning_outputs.py \
  --predictions "${OUTPUT_PATH}/planning_predictions.csv" \
  --ground-truth "${REPRO_ROOT}/UniAD_deploy/nuscenes_np/planning_ground_truth" \
  --output "${OUTPUT_PATH}/planning_metrics.json" \
  "${REFERENCE_ARGS[@]}"
