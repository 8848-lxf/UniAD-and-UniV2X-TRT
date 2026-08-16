#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env_modelopt.sh"

BUILDER=${UNIAD_TRT_BUILDER:-${REPRO_ROOT}/../../UniV2X/deploy_int8/tools/build_trt_engine.py}
ONNX_PATH=${UNIAD_FP_ONNX:-/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/onnx/uniad_tiny_imgx0.25_cp.repaired_simp.onnx}
PLUGIN_PATH=${UNIAD_TRT_PLUGIN:-/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/trt109_20260813/runtime_build/libuniad_plugin.so}
APP_PATH=${UNIAD_APP_PATH:-${REPRO_ROOT}/../runtime/inference_app/enqueueV3/build/uniad}
INPUT_PATH=${UNIAD_INPUT_PATH:-/home/lixingfeng/UniAD_examine/DL4AGX/AV-Solutions/uniad-trt/repro_20260806/UniAD_deploy/nuscenes_np/uniad_trt_input}
RUNTIME_WORKDIR=${UNIAD_RUNTIME_WORKDIR:-${REPRO_ROOT}/UniAD_deploy}
AUDIT_STAGE=${UNIAD_SEG_AUDIT_STAGE:-coarse}
if [[ -n "${UNIAD_SEG_AUDIT_ROOT:-}" ]]; then
  OUTPUT_ROOT=${UNIAD_SEG_AUDIT_ROOT}
elif [[ "${AUDIT_STAGE}" == coarse ]]; then
  OUTPUT_ROOT=/home/lixingfeng/uniad_trt_artifacts/fp16_seg_reverse_20260815
else
  OUTPUT_ROOT="/home/lixingfeng/uniad_trt_artifacts/fp16_seg_reverse_20260815_${AUDIT_STAGE}"
fi
GPU_LIST=${UNIAD_SEG_AUDIT_GPUS:-0,1,2,3,4,5,6,7}
NUM_FRAMES=${UNIAD_SEG_AUDIT_FRAMES:-40}
FIXED_TRACK_COUNT=${UNIAD_SEG_AUDIT_FIXED_TRACK_COUNT:-1150}
TRACK_MIN=${UNIAD_SEG_AUDIT_TRACK_MIN:-901}
TRACK_OPT=${UNIAD_SEG_AUDIT_TRACK_OPT:-901}
TRACK_MAX=${UNIAD_SEG_AUDIT_TRACK_MAX:-1150}
BUILD_ONLY=${UNIAD_SEG_AUDIT_BUILD_ONLY:-0}
EVALUATE_ONLY=${UNIAD_SEG_AUDIT_EVALUATE_ONLY:-0}
FORCE_REBUILD=${UNIAD_SEG_AUDIT_FORCE_REBUILD:-0}
FORCE_RERUN=${UNIAD_SEG_AUDIT_FORCE_RERUN:-0}

# Ordered from the shared occupancy feature into the final pre-threshold score.
# The lineage itself was traced backwards from seg_out; this forward order makes
# the first numerical error jump directly visible in the summary.
case "${AUDIT_STAGE}" in
  coarse)
    PROBE_SPECS=(
      "p00_input1743=input.1743"
      "p01_state=state"
      "p02_layer0_out=input.1807"
      "p03_layer0_fused=input.1819"
      "p04_layer1_out=input.1883"
      "p05_layer1_fused=input.1895"
      "p06_layer2_out=input.1959"
      "p07_layer2_fused=input.1971"
      "p08_layer3_out=input.2035"
      "p09_layer3_fused=input.2047"
      "p10_layer4_out=input.2111"
      "p11_layer4_fused=onnx::Unsqueeze_26264"
      "p12_dense_input=input.2123"
      "p13_dense_stage0=input.2143"
      "p14_dense_stage1=future_states"
      "p15_dense_crop=future_states.3"
      "p16_occ_einsum=onnx::Slice_26434"
      "p17_occ_slice=onnx::Sigmoid_26439"
      "p18_occ_sigmoid=onnx::Mul_26440"
      "p19_occ_mul=pred_ins_sigmoid"
      "p20_occ_concat=onnx::ReduceMax_26477"
      "p21_occ_score=onnx::Greater_26478"
    )
    ;;
  layer0)
    PROBE_SPECS=(
      "f00_state=state"
      "f01_gate_logits=onnx::Sigmoid_23764"
      "f02_gate_score=onnx::Less_23765"
      "f03_query=query.135"
      "f04_self_q=q.107"
      "f05_self_logits=attn.107"
      "f06_self_softmax=onnx::MatMul_23950"
      "f07_self_residual=input.1783"
      "f08_norm0=query.139"
      "f09_cross_q=q.111"
      "f10_cross_logits=attn.111"
      "f11_cross_softmax=onnx::MatMul_24104"
      "f12_cross_residual=input.1791"
      "f13_norm1=x.231"
      "f14_ffn_hidden=input.1795"
      "f15_ffn_relu=onnx::MatMul_24141"
      "f16_ffn_output=input.1799"
      "f17_ffn_residual=input.1803"
      "f18_norm2=onnx::Add_24155"
      "f19_layer0_out=input.1807"
    )
    ;;
  upstream)
    PROBE_SPECS=(
      "u00_track_scores=track_scores"
      "u01_active_index=active_index.7"
      "u02_valid_mask=onnx::Greater_23426"
      "u03_nonzero=onnx::NonZero_23428"
      "u04_out_track_query=out_track_query"
      "u05_ins_embed=ins_embed"
      "u06_scores1=scores.1"
      "u07_track_sigmoid_logits=onnx::Sigmoid_10611"
      "u08_track_sigmoid=onnx::ReduceMax_10612"
    )
    ;;
  *)
    echo "Unsupported UNIAD_SEG_AUDIT_STAGE: ${AUDIT_STAGE}" >&2
    exit 2
    ;;
esac
PRECISIONS=(fp32 fp16)

IFS=',' read -r -a GPUS <<< "${GPU_LIST}"
if [[ ${#GPUS[@]} -lt 1 ]]; then
  echo "UNIAD_SEG_AUDIT_GPUS must contain at least one GPU id" >&2
  exit 2
fi
if [[ "${BUILD_ONLY}" == 1 && "${EVALUATE_ONLY}" == 1 ]]; then
  echo "BUILD_ONLY and EVALUATE_ONLY cannot both be enabled" >&2
  exit 2
fi
for required_path in "${BUILDER}" "${ONNX_PATH}" "${PLUGIN_PATH}" \
  "${APP_PATH}" "${INPUT_PATH}/info.txt"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing layer-audit input: ${required_path}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}/engines" "${OUTPUT_ROOT}/reports" \
  "${OUTPUT_ROOT}/timing_cache" "${OUTPUT_ROOT}/layer_info" \
  "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/probe${NUM_FRAMES}"

TASK_KEYS=()
TASK_TENSORS=()
TASK_PRECISIONS=()
for specification in "${PROBE_SPECS[@]}"; do
  key=${specification%%=*}
  tensor=${specification#*=}
  for precision in "${PRECISIONS[@]}"; do
    TASK_KEYS+=("${key}")
    TASK_TENSORS+=("${tensor}")
    TASK_PRECISIONS+=("${precision}")
  done
done

build_task() {
  local gpu=$1
  local key=$2
  local tensor=$3
  local precision=$4
  local stem="${key}_${precision}"
  local engine="${OUTPUT_ROOT}/engines/${stem}.engine"
  local report="${OUTPUT_ROOT}/reports/${stem}.json"
  local cache="${OUTPUT_ROOT}/timing_cache/${stem}.cache"
  local layer_info="${OUTPUT_ROOT}/layer_info/${stem}.json"
  local log="${OUTPUT_ROOT}/logs/build_${stem}.log"

  if [[ "${FORCE_REBUILD}" == 1 ]]; then
    rm -f "${engine}" "${report}" "${cache}" "${layer_info}"
  fi
  if [[ ! -f "${engine}" || ! -f "${report}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" python "${BUILDER}" "${ONNX_PATH}" \
      --engine "${engine}" \
      --plugin "${PLUGIN_PATH}" \
      --precision "${precision}" \
      --timing-cache "${cache}" \
      --report "${report}" \
      --workspace-gib 8 \
      --track-min "${TRACK_MIN}" \
      --track-opt "${TRACK_OPT}" \
      --track-max "${TRACK_MAX}" \
      --optimization-level 3 \
      --mark-output-direct-alias "${tensor}=audit_out" \
      >"${log}" 2>&1
  fi
  if [[ ! -f "${layer_info}" ]]; then
    CUDA_VISIBLE_DEVICES="${gpu}" "${TRT_ROOT}/bin/trtexec" \
      --loadEngine="${engine}" \
      --staticPlugins="${PLUGIN_PATH}" \
      --profilingVerbosity=detailed \
      --dumpLayerInfo \
      --exportLayerInfo="${layer_info}" \
      --skipInference >>"${log}" 2>&1
  fi
}

evaluate_task() {
  local gpu=$1
  local key=$2
  local tensor=$3
  local precision=$4
  local stem="${key}_${precision}"
  local engine="${OUTPUT_ROOT}/engines/${stem}.engine"
  local result_root="${OUTPUT_ROOT}/probe${NUM_FRAMES}/${stem}"
  local log="${OUTPUT_ROOT}/logs/eval_${stem}_${NUM_FRAMES}.log"
  local output_file="${result_root}/audit_out.float32"

  if [[ ! -f "${engine}" ]]; then
    echo "Missing layer-audit engine: ${engine}" >&2
    return 2
  fi
  if [[ "${FORCE_RERUN}" == 1 ]]; then
    rm -rf "${result_root}"
  fi
  if [[ -f "${output_file}" && \
        -f "${result_root}/audit_out.float32.manifest.json" ]]; then
    return 0
  fi
  mkdir -p "${result_root}"
  (
    cd "${RUNTIME_WORKDIR}"
    CUDA_VISIBLE_DEVICES="${gpu}" \
    UNIAD_DISABLE_TEMPORAL_STATE=1 \
    UNIAD_COLLISION_OPTIMIZATION=0 \
    UNIAD_DUMP_AUDIT_OUTPUT=1 \
    "${APP_PATH}" \
      "${engine}" \
      "${PLUGIN_PATH}" \
      "${INPUT_PATH}" \
      "${result_root}" \
      "${NUM_FRAMES}" \
      "${result_root}/latency_metrics.json" \
      1 0 "${FIXED_TRACK_COUNT}" official_literal
  ) >"${log}" 2>&1
}

run_workers() {
  local action=$1
  local worker_pids=()
  local worker_count=${#GPUS[@]}
  local task_count=${#TASK_KEYS[@]}
  for worker_index in "${!GPUS[@]}"; do
    (
      task_index=${worker_index}
      while (( task_index < task_count )); do
        "${action}_task" \
          "${GPUS[worker_index]}" \
          "${TASK_KEYS[task_index]}" \
          "${TASK_TENSORS[task_index]}" \
          "${TASK_PRECISIONS[task_index]}"
        task_index=$((task_index + worker_count))
      done
    ) &
    worker_pids+=("$!")
  done
  local status=0
  for pid in "${worker_pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  return "${status}"
}

if [[ "${EVALUATE_ONLY}" != 1 ]]; then
  if ! run_workers build; then
    echo "One or more layer-audit builds failed; inspect ${OUTPUT_ROOT}/logs" >&2
    exit 1
  fi
fi
if [[ "${BUILD_ONLY}" == 1 ]]; then
  echo "Completed layer-audit builds at ${OUTPUT_ROOT}"
  exit 0
fi
if ! run_workers evaluate; then
  echo "One or more layer-audit evaluations failed; inspect ${OUTPUT_ROOT}/logs" >&2
  exit 1
fi

source "${REPRO_ROOT}/scripts/env.sh"
python "${REPRO_ROOT}/scripts/summarize_fp16_seg_reverse_layer_audit.py" \
  "${OUTPUT_ROOT}" \
  --stage "${AUDIT_STAGE}" \
  --frames "${NUM_FRAMES}" \
  --output "${OUTPUT_ROOT}/summary_${AUDIT_STAGE}_${NUM_FRAMES}.json" \
  --csv-output "${OUTPUT_ROOT}/summary_${AUDIT_STAGE}_${NUM_FRAMES}.csv"

echo "Completed seg_out reverse layer audit: ${OUTPUT_ROOT}/summary_${AUDIT_STAGE}_${NUM_FRAMES}.json"
