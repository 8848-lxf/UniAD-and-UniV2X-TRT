#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env_modelopt.sh"

OUTPUT_ROOT=${UNIAD_AB_ROOT:-/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/fp16_matmul_mul_ab_20260815}
PLUGIN_PATH=${UNIAD_TRT_PLUGIN:-/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/trt109_20260813/runtime_build/libuniad_plugin.so}
APP_PATH=${UNIAD_APP_PATH:-${REPRO_ROOT}/../runtime/inference_app/enqueueV3/build/uniad}
INPUT_PATH=${UNIAD_INPUT_PATH:-/home/lixingfeng/UniAD_examine/DL4AGX/AV-Solutions/uniad-trt/repro_20260806/UniAD_deploy/nuscenes_np/uniad_trt_input}
GROUND_TRUTH=${UNIAD_GROUND_TRUTH:-/home/lixingfeng/UniAD_examine/DL4AGX/AV-Solutions/uniad-trt/repro_20260806/UniAD_deploy/nuscenes_np/planning_ground_truth}
PYTORCH_RAW=${UNIAD_PYTORCH_RAW:-/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/trt109_20260813/cvjpeg_official_literal_full6018_20260813/pytorch_official_literal/forward_uniad_trt.npz.planning.csv}
GPU_LIST=${UNIAD_AB_GPUS:-0,1,2,3}
NUM_FRAMES=${UNIAD_AB_FRAMES:-200}
FIXED_TRACK_COUNT=${UNIAD_AB_FIXED_TRACK_COUNT:-1150}
RUNTIME_WORKDIR=${UNIAD_RUNTIME_WORKDIR:-${REPRO_ROOT}/UniAD_deploy}
if [[ -n "${UNIAD_PYTORCH_OCCUPANCY:-}" ]]; then
  PYTORCH_OCCUPANCY=${UNIAD_PYTORCH_OCCUPANCY}
elif [[ "${NUM_FRAMES}" == 200 ]]; then
  PYTORCH_OCCUPANCY=/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/trt109_20260814/probe200/pytorch_reference_first200.npz
elif [[ "${NUM_FRAMES}" == 6018 ]]; then
  PYTORCH_OCCUPANCY=/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/trt109_20260813/cvjpeg_official_literal_full6018_20260813/pytorch_official_literal/forward_uniad_trt.npz
else
  echo "Set UNIAD_PYTORCH_OCCUPANCY for NUM_FRAMES=${NUM_FRAMES}" >&2
  exit 2
fi

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "UNIAD_AB_GPUS must contain exactly four comma-separated GPU ids" >&2
  exit 2
fi

VARIANTS=(baseline matmul_fp32 mul_fp32 matmul_mul_fp32)
for required_path in "${PLUGIN_PATH}" "${APP_PATH}" "${INPUT_PATH}/info.txt" \
  "${GROUND_TRUTH}" "${PYTORCH_OCCUPANCY}" "${PYTORCH_RAW}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing A/B evaluation input: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}/probe${NUM_FRAMES}" "${OUTPUT_ROOT}/logs"

run_variant() {
  local variant=$1
  local gpu=$2
  local engine="${OUTPUT_ROOT}/engines/uniad_tiny_fp16_${variant}.engine"
  local result_root="${OUTPUT_ROOT}/probe${NUM_FRAMES}/${variant}"
  local log="${OUTPUT_ROOT}/logs/eval_${variant}_${NUM_FRAMES}.log"
  if [[ ! -f "${engine}" ]]; then
    echo "Missing A/B engine: ${engine}" >&2
    return 2
  fi
  mkdir -p "${result_root}"
  (
    cd "${RUNTIME_WORKDIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    UNIAD_COLLISION_OPTIMIZATION=1 \
    UNIAD_DUMP_OCCUPANCY=1 \
    "${APP_PATH}" \
      "${engine}" \
      "${PLUGIN_PATH}" \
      "${INPUT_PATH}" \
      "${result_root}" \
      "${NUM_FRAMES}" \
      "${result_root}/latency_metrics.json" \
      10 0 "${FIXED_TRACK_COUNT}" official_literal
  ) >"${log}" 2>&1
}

PIDS=()
for index in "${!VARIANTS[@]}"; do
  run_variant "${VARIANTS[index]}" "${GPUS[index]}" &
  PIDS+=("$!")
done
status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ ${status} -ne 0 ]]; then
  echo "One or more UniAD FP16 A/B evaluations failed" >&2
  exit ${status}
fi

source "${REPRO_ROOT}/scripts/env.sh"
for variant in "${VARIANTS[@]}"; do
  result_root="${OUTPUT_ROOT}/probe${NUM_FRAMES}/${variant}"
  python "${REPRO_ROOT}/scripts/compare_occupancy_audit.py" \
    --trt-packbits "${result_root}/seg_out.packbits" \
    --pytorch-audit "${PYTORCH_OCCUPANCY}" \
    --output "${result_root}/occupancy_vs_pytorch.json" \
    >"${OUTPUT_ROOT}/logs/compare_occupancy_${variant}_${NUM_FRAMES}.log" 2>&1
  python "${REPRO_ROOT}/UniAD_deploy/tools/evaluate_planning_outputs.py" \
    --predictions "${result_root}/planning_predictions_raw.csv" \
    --ground-truth "${GROUND_TRUTH}" \
    --output "${result_root}/planning_metrics_raw.json" \
    --reference-predictions "${PYTORCH_RAW}" \
    --prediction-protocol official_literal \
    --reference-protocol official_literal \
    >"${OUTPUT_ROOT}/logs/eval_raw_${variant}_${NUM_FRAMES}.log" 2>&1
  python "${REPRO_ROOT}/UniAD_deploy/tools/evaluate_planning_outputs.py" \
    --predictions "${result_root}/planning_predictions.csv" \
    --ground-truth "${GROUND_TRUTH}" \
    --output "${result_root}/planning_metrics_optimized.json" \
    --prediction-protocol official_literal \
    >"${OUTPUT_ROOT}/logs/eval_optimized_${variant}_${NUM_FRAMES}.log" 2>&1
done

python "${REPRO_ROOT}/scripts/summarize_fp16_matmul_mul_ab.py" \
  "${OUTPUT_ROOT}" \
  --frames "${NUM_FRAMES}" \
  --output "${OUTPUT_ROOT}/summary_${NUM_FRAMES}.json" \
  >"${OUTPUT_ROOT}/logs/summarize_${NUM_FRAMES}.log" 2>&1

echo "Completed UniAD FP16 MatMul/Mul A/B evaluation at ${OUTPUT_ROOT}/summary_${NUM_FRAMES}.json"
