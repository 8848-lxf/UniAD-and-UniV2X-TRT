#!/usr/bin/env bash

set -euo pipefail

REPRO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NUM_FRAMES=${1:-6018}

"${REPRO_ROOT}/scripts/run_trt_evaluation.sh" fp32 "${NUM_FRAMES}"
"${REPRO_ROOT}/scripts/run_trt_evaluation.sh" fp16 "${NUM_FRAMES}"
"${REPRO_ROOT}/scripts/run_trt_evaluation.sh" int8_eq_fp16 "${NUM_FRAMES}"
