#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 STAGE1_PID" >&2
  exit 2
fi

STAGE1_PID=$1
REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STATUS_LOG="${REPRO_ROOT}/logs/pipeline_status.log"
source "${REPRO_ROOT}/scripts/env.sh"

mark_stage() {
  printf '%s\t%s\n' "$(date --iso-8601=seconds)" "$1" >> "${STATUS_LOG}"
}

validate_checkpoint() {
  local checkpoint=$1
  if [[ ! -s "${checkpoint}" ]]; then
    echo "Missing or empty checkpoint: ${checkpoint}" >&2
    return 1
  fi
  python -c 'import sys, torch; c=torch.load(sys.argv[1], map_location="cpu"); assert isinstance(c, dict) and isinstance(c.get("state_dict"), dict) and c["state_dict"]' "${checkpoint}"
}

link_checkpoint() {
  local target=$1
  local link_path=$2
  if [[ -L "${link_path}" ]]; then
    if [[ "$(readlink -f "${link_path}")" != "$(readlink -f "${target}")" ]]; then
      echo "Checkpoint link points to an unexpected target: ${link_path}" >&2
      return 1
    fi
  elif [[ -e "${link_path}" ]]; then
    echo "Refusing to replace existing checkpoint path: ${link_path}" >&2
    return 1
  else
    ln -s "${target}" "${link_path}"
  fi
}

mark_stage "waiting_for_stage1_pid_${STAGE1_PID}"
while kill -0 "${STAGE1_PID}" 2>/dev/null; do
  sleep 60
done

STAGE1_CHECKPOINT="${REPRO_ROOT}/artifacts/stage1/epoch_6.pth"
validate_checkpoint "${STAGE1_CHECKPOINT}"
link_checkpoint \
  "${STAGE1_CHECKPOINT}" \
  "${REPRO_ROOT}/artifacts/checkpoints/tiny_imgx0.25_track_map.pth"
mark_stage "stage1_checkpoint_valid"

mark_stage "stage2_training_started"
"${REPRO_ROOT}/scripts/run_stage2.sh" > "${REPRO_ROOT}/logs/stage2_train_20260806.log" 2>&1
STAGE2_CHECKPOINT="${REPRO_ROOT}/artifacts/stage2/epoch_20.pth"
validate_checkpoint "${STAGE2_CHECKPOINT}"
link_checkpoint \
  "${STAGE2_CHECKPOINT}" \
  "${REPRO_ROOT}/artifacts/checkpoints/tiny_imgx0.25_e2e_ep20.pth"
mark_stage "stage2_checkpoint_valid"

mark_stage "pytorch_evaluation_started"
"${REPRO_ROOT}/scripts/run_pytorch_evaluation.sh" > "${REPRO_ROOT}/logs/pytorch_evaluation.log" 2>&1
mark_stage "pytorch_evaluation_complete"

mark_stage "onnx_export_started"
"${REPRO_ROOT}/scripts/run_export_onnx.sh" > "${REPRO_ROOT}/logs/onnx_export.log" 2>&1
mark_stage "onnx_export_complete"

mark_stage "calibration_collection_started"
"${REPRO_ROOT}/scripts/run_prepare_calibration.sh" > "${REPRO_ROOT}/logs/calibration_collection.log" 2>&1
mark_stage "calibration_collection_complete"

mark_stage "explicit_quantization_started"
"${REPRO_ROOT}/scripts/run_quantize_onnx.sh" > "${REPRO_ROOT}/logs/explicit_quantization.log" 2>&1
mark_stage "explicit_quantization_complete"

mark_stage "quantization_validation_started"
"${REPRO_ROOT}/scripts/run_validate_quant_artifacts.sh" > "${REPRO_ROOT}/logs/quantization_validation.log" 2>&1
mark_stage "quantization_validation_complete"

mark_stage "engine_build_started"
"${REPRO_ROOT}/scripts/run_build_engines.sh" > "${REPRO_ROOT}/logs/engine_build.log" 2>&1
mark_stage "engine_build_complete"

mark_stage "tensorrt_evaluations_started"
"${REPRO_ROOT}/scripts/run_all_trt_evaluations.sh" 6018 > "${REPRO_ROOT}/logs/tensorrt_evaluations.log" 2>&1
mark_stage "tensorrt_evaluations_complete"

mark_stage "final_report_started"
python "${REPRO_ROOT}/scripts/generate_final_report.py" > "${REPRO_ROOT}/logs/final_report.log" 2>&1
mark_stage "final_report_complete"
mark_stage "pipeline_complete"
