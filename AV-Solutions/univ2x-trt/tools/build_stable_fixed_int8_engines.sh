#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/lixingfeng/UniAD_examine/UniV2X
CONDA_ROOT=/home/lixingfeng/anaconda3
TRT_ROOT=/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118
TRT_LIB=${TRT_ROOT}/targets/x86_64-linux-gnu/lib
ARTIFACT_ROOT=${ROOT}/deploy_int8/artifacts/camera_repeat_fix
FIX_ROOT=${ARTIFACT_ROOT}/numerical_stability_fix
PLUGIN=${ROOT}/deploy_int8/plugins_msda_fp32accum/build_univ2x_fp32accum/lib_uniad_plugins_trt10.9_x86_cu118.so
BUILDER=${ROOT}/deploy_int8/tools/build_trt_engine.py

# Conda's compiler activation hooks legitimately probe unset SYS_* variables.
set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate modelopt
set -u
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
export LD_LIBRARY_PATH=${TRT_LIB}:${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}
export PYTHONUNBUFFERED=1

for tool in python nvcc g++; do
    resolved=$(command -v "${tool}")
    case "${resolved}" in
        "${CONDA_PREFIX}"/*) ;;
        *) echo "Refusing non-Conda toolchain executable: ${tool}=${resolved}" >&2; exit 1 ;;
    esac
done

python - <<'PY'
import tensorrt as trt
assert trt.__version__ == "10.9.0.34", trt.__version__
print("TensorRT", trt.__version__)
PY

mkdir -p "${FIX_ROOT}/engines" "${FIX_ROOT}/timing_cache" \
    "${FIX_ROOT}/reports" "${FIX_ROOT}/logs"

initialize_cache() {
    local source_cache=$1
    local target_cache=$2
    if [[ ! -e "${target_cache}" ]]; then
        cp "${source_cache}" "${target_cache}"
    fi
}

run_build() {
    local engine=$1
    local report=$2
    local log=$3
    shift 3
    if [[ -s "${engine}" && -s "${report}" ]]; then
        echo "Existing completed build retained: ${engine}"
        return
    fi
    printf 'Running:'
    printf ' %q' "$@"
    printf '\n'
    "$@" 2>&1 | tee "${log}"
}

INFRA_ONNX=${FIX_ROOT}/onnx/infrastructure_int8_qdq_numerically_stable.onnx
INFRA_ENGINE=${FIX_ROOT}/engines/infrastructure_runtime_int8_stable_fixed960.engine
INFRA_CACHE=${FIX_ROOT}/timing_cache/infrastructure_runtime_int8_stable_fixed960.cache
INFRA_REPORT=${FIX_ROOT}/reports/infrastructure_runtime_int8_stable_fixed960_build.json
INFRA_LOG=${FIX_ROOT}/logs/infrastructure_runtime_int8_stable_fixed960_build.log
initialize_cache \
    "${ARTIFACT_ROOT}/timing_cache/infrastructure_runtime_int8.cache" \
    "${INFRA_CACHE}"

infra_cmd=(
    python "${BUILDER}" "${INFRA_ONNX}"
    --engine "${INFRA_ENGINE}"
    --plugin "${PLUGIN}"
    --precision int8
    --timing-cache "${INFRA_CACHE}"
    --report "${INFRA_REPORT}"
    --workspace-gib 8
    --optimization-level 3
    --track-min 960 --track-opt 960 --track-max 960
    --stabilize-inverse-sigmoid
    --stabilize-layernorm
    --force-fp32-layer-regex '^Identity_29235$'
    --force-fp32-layer-regex '_(565|567|568|572|573|684|893|894|895|896|897|904)$'
    --force-fp32-layer-regex '^MultiScaleDeformableAttnTRT_(3120|3215|3334|3429|3548|3643|3762|3857|3976|4071|4190|4285)$'
    --fp32-lineage-depth 120
)

infra_fp32_lineages=(
    value reference_points sampling_offsets attention_weights
    value.3 reference_points.3 sampling_offsets.3 attention_weights.3
    value.7 sampling_offsets.7 attention_weights.7
    value.11 reference_points.7 sampling_offsets.11 attention_weights.11
    value.15 sampling_offsets.15 attention_weights.15
    value.19 reference_points.11 sampling_offsets.19 attention_weights.19
    value.23 sampling_offsets.23 attention_weights.23
    value.27 reference_points.15 sampling_offsets.27 attention_weights.27
    value.31 sampling_offsets.31 attention_weights.31
    value.35 reference_points.19 sampling_offsets.35 attention_weights.35
    value.39 sampling_offsets.39 attention_weights.39
    value.43 reference_points.23 sampling_offsets.43 attention_weights.43
)
for tensor in "${infra_fp32_lineages[@]}"; do
    infra_cmd+=(--force-fp32-tensor-lineage "${tensor}")
done
run_build "${INFRA_ENGINE}" "${INFRA_REPORT}" "${INFRA_LOG}" "${infra_cmd[@]}"

EGO_ONNX=${FIX_ROOT}/onnx/ego_int8_qdq_agent_stable.onnx
EGO_ENGINE=${FIX_ROOT}/engines/ego_runtime_int8_stable_fixed960_64.engine
EGO_CACHE=${FIX_ROOT}/timing_cache/ego_runtime_int8_stable_fixed960_64.cache
EGO_REPORT=${FIX_ROOT}/reports/ego_runtime_int8_stable_fixed960_64_build.json
EGO_LOG=${FIX_ROOT}/logs/ego_runtime_int8_stable_fixed960_64_build.log
initialize_cache \
    "${ARTIFACT_ROOT}/timing_cache/ego_runtime_int8.cache" \
    "${EGO_CACHE}"

ego_cmd=(
    python "${BUILDER}" "${EGO_ONNX}"
    --engine "${EGO_ENGINE}"
    --plugin "${PLUGIN}"
    --precision int8
    --timing-cache "${EGO_CACHE}"
    --report "${EGO_REPORT}"
    --workspace-gib 8
    --optimization-level 3
    --track-min 960 --track-opt 960 --track-max 960
    --coop-min 64 --coop-opt 64 --coop-max 64
    --stabilize-inverse-sigmoid
    --stabilize-layernorm
)
run_build "${EGO_ENGINE}" "${EGO_REPORT}" "${EGO_LOG}" "${ego_cmd[@]}"
