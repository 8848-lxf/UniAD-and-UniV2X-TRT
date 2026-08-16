#!/usr/bin/env bash
set -euo pipefail

UNIV2X_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CARLA_WORKSPACE="/home/lixingfeng/UniAD_examine/Carla"
CARLA_EGG="${CARLA_WORKSPACE}/carla-0.9.10.1/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"
PYTHON_BIN="${CARLA_WORKSPACE}/.conda/carla0910/bin/python"
CARLA_PORT="${CARLA_PORT:-2010}"
FRAMES="${FRAMES:-20}"
VEHICLES="${VEHICLES:-20}"
OUTPUT_DIR="${1:-${UNIV2X_ROOT}/deploy_int8/artifacts/carla/capture_$(date +%Y%m%d_%H%M%S)}"

if [[ ! -f "${CARLA_EGG}" ]]; then
    echo "CARLA PythonAPI egg not found: ${CARLA_EGG}" >&2
    exit 1
fi

env PYTHONPATH="${CARLA_EGG}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" \
    "${UNIV2X_ROOT}/deploy_int8/carla/capture_univ2x_sequence.py" \
    --port "${CARLA_PORT}" \
    --frames "${FRAMES}" \
    --vehicles "${VEHICLES}" \
    --output-dir "${OUTPUT_DIR}"

echo "UniV2X CARLA capture complete: ${OUTPUT_DIR}"
