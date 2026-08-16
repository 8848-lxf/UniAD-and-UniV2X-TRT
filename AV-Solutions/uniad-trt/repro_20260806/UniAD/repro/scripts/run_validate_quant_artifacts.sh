#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPRO_ROOT}/scripts/env_modelopt.sh"
ARTIFACT_ROOT=${ARTIFACT_ROOT:-${REPRO_ROOT}/artifacts}

exec python "${REPRO_ROOT}/scripts/validate_quant_artifacts.py" \
  --fp-onnx "${ARTIFACT_ROOT}/onnx/uniad_tiny_imgx0.25_cp.repaired.onnx" \
  --quant-onnx "${ARTIFACT_ROOT}/onnx/uniad_tiny_int8_eq_dq_only.onnx" \
  --calibration "${ARTIFACT_ROOT}/calibration/calib_data_shape0_901.npz" \
  --output "${ARTIFACT_ROOT}/onnx/quantization_inspection.json"
