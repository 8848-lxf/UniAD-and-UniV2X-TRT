#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPOSITORY_ROOT=$(cd "${REPRO_ROOT}/../.." && pwd)
source "${REPRO_ROOT}/scripts/env_trt107.sh"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 QUANTIZED_ONNX OUTPUT_ROOT" >&2
  exit 2
fi

ONNX_PATH=$(realpath "$1")
OUTPUT_ROOT=$(realpath -m "$2")
UNIAD_GPU=${UNIAD_GPU:-5}
TRACK_COUNT=${TRACK_COUNT:-1600}
BUILDER=${UNIAD_TRT_BUILDER:-${REPOSITORY_ROOT}/UniV2X/deploy_int8/tools/build_trt_engine.py}
PLUGIN=${UNIAD_TRT107_BUILD_PLUGIN:-/home/lixingfeng/UniAD_examine/UniV2X/deploy_int8/artifacts/plugins/trt107/libuniad_plugin_trt107_abi_test.so}

for required_path in "${ONNX_PATH}" "${BUILDER}" "${PLUGIN}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing exact-10.7 build input: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}/engines" "${OUTPUT_ROOT}/reports" \
  "${OUTPUT_ROOT}/timing_cache" "${OUTPUT_ROOT}/logs"
ENGINE_PATH="${OUTPUT_ROOT}/engines/uniad_tiny_int8_eq_fp16_static${TRACK_COUNT}.engine"
REPORT_PATH="${OUTPUT_ROOT}/reports/uniad_tiny_int8_eq_fp16_static${TRACK_COUNT}.json"
CACHE_PATH="${OUTPUT_ROOT}/timing_cache/uniad_tiny_int8_eq_fp16_static${TRACK_COUNT}.cache"

if [[ -e "${ENGINE_PATH}" || -e "${CACHE_PATH}" || -e "${REPORT_PATH}" ]]; then
  echo "Refusing to overwrite an existing hardware-specific artifact under ${OUTPUT_ROOT}" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${UNIAD_GPU}"
python "${BUILDER}" "${ONNX_PATH}" \
  --engine "${ENGINE_PATH}" \
  --plugin "${PLUGIN}" \
  --precision int8 \
  --timing-cache "${CACHE_PATH}" \
  --report "${REPORT_PATH}" \
  --workspace-gib 8 \
  --track-min "${TRACK_COUNT}" \
  --track-opt "${TRACK_COUNT}" \
  --track-max "${TRACK_COUNT}" \
  --optimization-level 3

sha256sum "${ONNX_PATH}" "${PLUGIN}" "${ENGINE_PATH}" > \
  "${OUTPUT_ROOT}/reports/artifact_sha256.txt"
