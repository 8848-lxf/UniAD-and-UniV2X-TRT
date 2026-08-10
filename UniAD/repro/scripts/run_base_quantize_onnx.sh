#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env_modelopt.sh"

BASE_ROOT=${BASE_ROOT:-${REPRO_ROOT}/artifacts/uniad_base_e2e}
UNIAD_GPU=${UNIAD_GPU:-6}
CALIBRATION_METHOD=${CALIBRATION_METHOD:-entropy}
ONNX_PATH=${1:-${BASE_ROOT}/onnx/uniad_base_e2e_dcn_plugin.repaired.onnx}
CALIBRATION_PATH=${2:-${BASE_ROOT}/calibration/train_calib_shape0_901.npz}
OUTPUT_PATH=${3:-${BASE_ROOT}/onnx/uniad_base_e2e_int8_eq_dq_only.onnx}
PLUGIN_PATH="${REPRO_ROOT}/UniAD_deploy/plugins/lib/lib_uniad_plugins_trt10.9_x86_cu118.so"

for required_path in "${ONNX_PATH}" "${CALIBRATION_PATH}" "${PLUGIN_PATH}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing UniAD-base quantization input: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "$(dirname "${OUTPUT_PATH}")"
export CUDA_VISIBLE_DEVICES="${UNIAD_GPU}"

CALIBRATION_SHAPES='prev_track_intances0:901x512,prev_track_intances1:901x3,prev_track_intances3:901,prev_track_intances4:901,prev_track_intances5:901,prev_track_intances6:901,prev_track_intances8:901,prev_track_intances9:901x10,prev_track_intances11:901x4x256,prev_track_intances12:901x4,prev_track_intances13:901,prev_timestamp:1,prev_l2g_r_mat:1x3x3,prev_l2g_t:1x3,prev_bev:40000x1x256,timestamp:1,l2g_r_mat:1x3x3,l2g_t:1x3,img:1x6x3x928x1600,img_metas_can_bus:18,img_metas_lidar2img:1x6x4x4,command:1,use_prev_bev:1,max_obj_id:1'

exec python -m modelopt.onnx.quantization \
  --onnx_path="${ONNX_PATH}" \
  --quantize_mode int8 \
  --calibration_method="${CALIBRATION_METHOD}" \
  --trt_plugins="${PLUGIN_PATH}" \
  --calibration_eps trt cuda:0 cpu \
  --calibration_shapes="${CALIBRATION_SHAPES}" \
  --simplify \
  --op_types_to_exclude MatMul \
  --calibration_data_path="${CALIBRATION_PATH}" \
  --output_path="${OUTPUT_PATH}" \
  --dq_only
