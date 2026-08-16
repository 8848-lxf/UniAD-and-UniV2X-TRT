#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env_modelopt.sh"

ARTIFACT_ROOT=${ARTIFACT_ROOT:-${REPRO_ROOT}/artifacts}
UNIAD_GPU=${UNIAD_GPU:-0}
ITERATIONS=${ITERATIONS:-100}
ENGINE_SET=${ENGINE_SET:-fp32,fp16,int8_eq_fp16}
PLUGIN_PATH="${REPRO_ROOT}/package/uniad-trt/inference_app/enqueueV3/build/libuniad_plugin.so"
OUTPUT_DIR="${ARTIFACT_ROOT}/evaluation/trtexec"
SHAPES='prev_track_intances0:901x512,prev_track_intances1:901x3,prev_track_intances3:901,prev_track_intances4:901,prev_track_intances5:901,prev_track_intances6:901,prev_track_intances8:901,prev_track_intances9:901x10,prev_track_intances11:901x4x256,prev_track_intances12:901x4,prev_track_intances13:901'

mkdir -p "${OUTPUT_DIR}"
export CUDA_VISIBLE_DEVICES="${UNIAD_GPU}"

for precision in fp32 fp16 int8_eq_fp16; do
  if [[ ",${ENGINE_SET}," != *",${precision},"* ]]; then
    continue
  fi
  engine_path="${ARTIFACT_ROOT}/engines/uniad_tiny_${precision}.engine"
  if [[ ! -f "${engine_path}" ]]; then
    echo "Missing benchmark engine: ${engine_path}" >&2
    exit 2
  fi

  "${TRT_ROOT}/bin/trtexec" \
    --loadEngine="${engine_path}" \
    --staticPlugins="${PLUGIN_PATH}" \
    --shapes="${SHAPES}" \
    --iterations="${ITERATIONS}" \
    --duration=0 \
    --warmUp=200 \
    --exportTimes="${OUTPUT_DIR}/${precision}_times.json" \
    2>&1 | tee "${OUTPUT_DIR}/${precision}.log"
done
