#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env_modelopt.sh"

BUILDER=${UNIAD_TRT_BUILDER:-${REPRO_ROOT}/../../UniV2X/deploy_int8/tools/build_trt_engine.py}
ONNX_PATH=${UNIAD_FP_ONNX:-/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/onnx/uniad_tiny_imgx0.25_cp.repaired_simp.onnx}
PLUGIN_PATH=${UNIAD_TRT_PLUGIN:-/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/trt109_20260813/runtime_build/libuniad_plugin.so}
OUTPUT_ROOT=${UNIAD_AB_ROOT:-/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/fp16_matmul_mul_ab_20260815}
GPU_LIST=${UNIAD_AB_GPUS:-0,1,2,3}
TRACK_MIN=${UNIAD_AB_TRACK_MIN:-901}
TRACK_OPT=${UNIAD_AB_TRACK_OPT:-901}
TRACK_MAX=${UNIAD_AB_TRACK_MAX:-1150}

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ ${#GPUS[@]} -ne 4 ]]; then
  echo "UNIAD_AB_GPUS must contain exactly four comma-separated GPU ids" >&2
  exit 2
fi

for required_path in "${BUILDER}" "${ONNX_PATH}" "${PLUGIN_PATH}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing A/B build input: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}/engines" "${OUTPUT_ROOT}/reports" \
  "${OUTPUT_ROOT}/timing_cache" "${OUTPUT_ROOT}/logs" \
  "${OUTPUT_ROOT}/layer_info"

VARIANTS=(baseline matmul_fp32 mul_fp32 matmul_mul_fp32)

build_variant() {
  local variant=$1
  local gpu=$2
  shift 2
  local engine="${OUTPUT_ROOT}/engines/uniad_tiny_fp16_${variant}.engine"
  local report="${OUTPUT_ROOT}/reports/${variant}.json"
  local cache="${OUTPUT_ROOT}/timing_cache/${variant}.cache"
  local log="${OUTPUT_ROOT}/logs/build_${variant}.log"
  local layer_info="${OUTPUT_ROOT}/layer_info/${variant}.json"

  if [[ -f "${engine}" && -f "${report}" && "${UNIAD_AB_FORCE_REBUILD:-0}" != 1 ]]; then
    echo "Reusing completed ${variant}: ${engine}"
  else
    rm -f "${engine}" "${report}" "${cache}" "${layer_info}"
    CUDA_VISIBLE_DEVICES="${gpu}" python "${BUILDER}" "${ONNX_PATH}" \
      --engine "${engine}" \
      --plugin "${PLUGIN_PATH}" \
      --precision fp16 \
      --timing-cache "${cache}" \
      --report "${report}" \
      --workspace-gib 8 \
      --track-min "${TRACK_MIN}" \
      --track-opt "${TRACK_OPT}" \
      --track-max "${TRACK_MAX}" \
      --optimization-level 3 \
      "$@" >"${log}" 2>&1
  fi

  CUDA_VISIBLE_DEVICES="${gpu}" "${TRT_ROOT}/bin/trtexec" \
    --loadEngine="${engine}" \
    --staticPlugins="${PLUGIN_PATH}" \
    --profilingVerbosity=detailed \
    --dumpLayerInfo \
    --exportLayerInfo="${layer_info}" \
    --skipInference >>"${log}" 2>&1
}

build_variant "${VARIANTS[0]}" "${GPUS[0]}" &
PIDS=("$!")
build_variant "${VARIANTS[1]}" "${GPUS[1]}" \
  --force-fp32-operator MatMul &
PIDS+=("$!")
build_variant "${VARIANTS[2]}" "${GPUS[2]}" \
  --force-fp32-operator Mul &
PIDS+=("$!")
build_variant "${VARIANTS[3]}" "${GPUS[3]}" \
  --force-fp32-operator MatMul \
  --force-fp32-operator Mul &
PIDS+=("$!")

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ ${status} -ne 0 ]]; then
  echo "One or more UniAD FP16 A/B builds failed; inspect ${OUTPUT_ROOT}/logs" >&2
  exit ${status}
fi

echo "Completed UniAD FP16 MatMul/Mul A/B builds at ${OUTPUT_ROOT}"
