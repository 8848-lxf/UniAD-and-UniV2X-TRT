#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 CAPTURE_DIR {fp32|fp16|int8} [OUTPUT_JSON]" >&2
    exit 2
fi

UNIV2X_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CAPTURE_DIR="$(readlink -f "$1")"
PRECISION_KEY="$2"
GPU="${UNIV2X_CARLA_GPU:-7}"
PYTHON_BIN="/home/lixingfeng/anaconda3/envs/univ2x_trt_runtime/bin/python"
TRT_LIB="/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118/targets/x86_64-linux-gnu/lib"
ENV_LIB="/home/lixingfeng/anaconda3/envs/univ2x_trt_runtime/lib"
PLUGIN="${UNIV2X_ROOT}/deploy_int8/plugins_msda_fp32accum/build_univ2x_fp32accum/lib_uniad_plugins_trt10.9_x86_cu118.so"
TEMPLATE_DIR="${UNIV2X_ROOT}/deploy_int8/artifacts/export_inputs/frame0"
CAMERA_FIX_DIR="${UNIV2X_ROOT}/deploy_int8/artifacts/camera_repeat_fix"
STABLE_DIR="${CAMERA_FIX_DIR}/numerical_stability_fix"
EXTRA_ARGS=()

case "${PRECISION_KEY}" in
    fp32)
        PRECISION="FP32"
        INFRA_ENGINE="${UNIV2X_INFRA_ENGINE:-${CAMERA_FIX_DIR}/engines/infrastructure_runtime_fp32.engine}"
        EGO_ENGINE="${UNIV2X_EGO_ENGINE:-${CAMERA_FIX_DIR}/engines/ego_runtime_fp32.engine}"
        ;;
    fp16)
        PRECISION="FP16"
        INFRA_ENGINE="${UNIV2X_INFRA_ENGINE:-${STABLE_DIR}/engines/infrastructure_runtime_fp16_v11_timestamp_bev_fp32.engine}"
        EGO_ENGINE="${UNIV2X_EGO_ENGINE:-${STABLE_DIR}/engines/ego_runtime_fp16_stable_v13_agent_reference.engine}"
        ;;
    int8)
        PRECISION="INT8-EQ+FP16"
        INFRA_ENGINE="${UNIV2X_INFRA_ENGINE:-${STABLE_DIR}/engines/infrastructure_runtime_int8_stable_fixed960.engine}"
        EGO_ENGINE="${UNIV2X_EGO_ENGINE:-${STABLE_DIR}/engines/ego_runtime_int8_stable_fixed960_64.engine}"
        ;;
    *)
        echo "unknown precision: ${PRECISION_KEY}" >&2
        exit 2
        ;;
esac

OUTPUT_JSON="${3:-${UNIV2X_ROOT}/deploy_int8/artifacts/carla/replay_${PRECISION_KEY}_$(date +%Y%m%d_%H%M%S).json}"
for required in "${CAPTURE_DIR}/manifest.json" "${INFRA_ENGINE}" "${EGO_ENGINE}" "${PLUGIN}"; do
    if [[ ! -f "${required}" ]]; then
        echo "required artifact not found: ${required}" >&2
        exit 1
    fi
done

env CUDA_VISIBLE_DEVICES="${GPU}" \
    LD_LIBRARY_PATH="${TRT_LIB}:${ENV_LIB}" \
    "${PYTHON_BIN}" "${UNIV2X_ROOT}/deploy_int8/carla/eval_univ2x_carla_replay.py" \
    "${CAPTURE_DIR}" "${INFRA_ENGINE}" "${EGO_ENGINE}" \
    --plugin "${PLUGIN}" \
    --precision "${PRECISION}" \
    --template-input-dir "${TEMPLATE_DIR}" \
    --fixed-track-count 960 \
    --fixed-coop-count 64 \
    --output "${OUTPUT_JSON}" \
    "${EXTRA_ARGS[@]}"

echo "UniV2X CARLA replay complete: ${OUTPUT_JSON}"
