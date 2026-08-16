#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env_modelopt.sh"

BASE_ROOT=${BASE_ROOT:-${REPRO_ROOT}/artifacts/uniad_base_e2e}
UNIAD_GPU=${UNIAD_GPU:-6}
ENGINE_SET=${ENGINE_SET:-fp32,fp16,int8_eq_fp16}
FP_ONNX=${1:-${BASE_ROOT}/onnx/uniad_base_e2e_dcn_plugin.repaired.onnx}
INT8_ONNX=${2:-${BASE_ROOT}/onnx/uniad_base_e2e_int8_eq_dq_only.onnx}
APP_ROOT=${UNIAD_APP_ROOT:-${REPRO_ROOT}/../runtime/inference_app/enqueueV3}
PLUGIN_PATH="${APP_ROOT}/build_base/libuniad_plugin.so"
ENGINE_DIR="${BASE_ROOT}/engines"
CACHE_DIR="${BASE_ROOT}/timing_cache"

required_paths=("${PLUGIN_PATH}")
if [[ ",${ENGINE_SET}," == *,fp32,* || ",${ENGINE_SET}," == *,fp16,* ]]; then
  required_paths+=("${FP_ONNX}")
fi
if [[ ",${ENGINE_SET}," == *,int8_eq_fp16,* ]]; then
  required_paths+=("${INT8_ONNX}")
fi
for required_path in "${required_paths[@]}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing UniAD-base engine input: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${ENGINE_DIR}" "${CACHE_DIR}"
export CUDA_VISIBLE_DEVICES="${UNIAD_GPU}"

MIN=901
OPT=901
MAX=1150
DYNAMIC_SHAPES='prev_track_intances0:901x512,prev_track_intances1:901x3,prev_track_intances3:901,prev_track_intances4:901,prev_track_intances5:901,prev_track_intances6:901,prev_track_intances8:901,prev_track_intances9:901x10,prev_track_intances11:901x4x256,prev_track_intances12:901x4,prev_track_intances13:901'
MIN_SHAPES=${DYNAMIC_SHAPES//901/${MIN}}
OPT_SHAPES=${DYNAMIC_SHAPES//901/${OPT}}
MAX_SHAPES=${DYNAMIC_SHAPES//901/${MAX}}

build_engine() {
  local onnx_path=$1
  local engine_path=$2
  local cache_path=$3
  shift 3
  "${TRT_ROOT}/bin/trtexec" \
    --onnx="${onnx_path}" \
    --saveEngine="${engine_path}" \
    --staticPlugins="${PLUGIN_PATH}" \
    --profilingVerbosity=detailed \
    --tacticSources=+CUBLAS \
    --minShapes="${MIN_SHAPES}" \
    --optShapes="${OPT_SHAPES}" \
    --maxShapes="${MAX_SHAPES}" \
    --memPoolSize=workspace:8G \
    --timingCacheFile="${cache_path}" \
    --skipInference \
    "$@"
}

if [[ ",${ENGINE_SET}," == *,fp32,* ]]; then
  build_engine "${FP_ONNX}" "${ENGINE_DIR}/uniad_base_fp32.engine" "${CACHE_DIR}/fp32.cache"
fi
if [[ ",${ENGINE_SET}," == *,fp16,* ]]; then
  build_engine "${FP_ONNX}" "${ENGINE_DIR}/uniad_base_fp16.engine" "${CACHE_DIR}/fp16.cache" --fp16
fi
if [[ ",${ENGINE_SET}," == *,int8_eq_fp16,* ]]; then
  build_engine "${INT8_ONNX}" "${ENGINE_DIR}/uniad_base_int8_eq_fp16.engine" "${CACHE_DIR}/int8_eq_fp16.cache" --best
fi
