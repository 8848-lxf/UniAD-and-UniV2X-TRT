#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env_modelopt.sh"
UNIAD_GPU=${UNIAD_GPU:-0}
CALIBRATION_METHOD=${CALIBRATION_METHOD:-entropy}
UNIAD_SIMPLIFY=${UNIAD_SIMPLIFY:-1}
UNIAD_NODES_TO_EXCLUDE=${UNIAD_NODES_TO_EXCLUDE:-}

ONNX_PATH=${1:-${REPRO_ROOT}/artifacts/onnx/uniad_tiny_imgx0.25_cp.repaired.onnx}
CALIBRATION_PATH=${2:-${REPRO_ROOT}/artifacts/calibration/calib_data_shape0_901.npz}
OUTPUT_PATH=${3:-${REPRO_ROOT}/artifacts/onnx/uniad_tiny_int8_eq_dq_only.onnx}
PLUGIN_PATH="${REPRO_ROOT}/UniAD_deploy/plugins/lib/lib_uniad_plugins_trt10.9_x86_cu118.so"

for required_path in "${ONNX_PATH}" "${CALIBRATION_PATH}" "${PLUGIN_PATH}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing quantization input: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "${OUTPUT_PATH}")"
export CUDA_VISIBLE_DEVICES="${UNIAD_GPU}"
export UNIAD_CALIBRATION_PROVIDER_REPORT=${UNIAD_CALIBRATION_PROVIDER_REPORT:-"${OUTPUT_PATH%.onnx}.calibration_provider.json"}

CALIBRATION_SHAPES='prev_track_intances0:901x512,prev_track_intances1:901x3,prev_track_intances3:901,prev_track_intances4:901,prev_track_intances5:901,prev_track_intances6:901,prev_track_intances8:901,prev_track_intances9:901x10,prev_track_intances11:901x4x256,prev_track_intances12:901x4,prev_track_intances13:901,prev_timestamp:1,prev_l2g_r_mat:1x3x3,prev_l2g_t:1x3,prev_bev:2500x1x256,timestamp:1,l2g_r_mat:1x3x3,l2g_t:1x3,img:1x6x3x256x416,img_metas_can_bus:18,img_metas_lidar2img:1x6x4x4,command:1,use_prev_bev:1,max_obj_id:1'
SIMPLIFY_ARGS=()
if [[ "${UNIAD_SIMPLIFY}" == "1" ]]; then
  SIMPLIFY_ARGS=(--simplify)
fi
NODE_EXCLUDE_ARGS=()
if [[ -n "${UNIAD_NODES_TO_EXCLUDE}" ]]; then
  read -r -a NODE_EXCLUDES <<< "${UNIAD_NODES_TO_EXCLUDE}"
  NODE_EXCLUDE_ARGS=(--nodes_to_exclude "${NODE_EXCLUDES[@]}")
fi

exec python "${REPRO_ROOT}/scripts/quantize_onnx_fixed_calibration.py" \
  --onnx_path="${ONNX_PATH}" \
  --quantize_mode int8 \
  --calibration_method="${CALIBRATION_METHOD}" \
  --trt_plugins="${PLUGIN_PATH}" \
  --calibration_eps trt cuda:0 cpu \
  --calibration_shapes="${CALIBRATION_SHAPES}" \
  "${SIMPLIFY_ARGS[@]}" \
  --op_types_to_exclude MatMul \
  "${NODE_EXCLUDE_ARGS[@]}" \
  --calibration_data_path="${CALIBRATION_PATH}" \
  --output_path="${OUTPUT_PATH}" \
  --dq_only
