#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
    echo "usage: $0 PRECISION INFRA_ENGINE EGO_ENGINE ROUTE_XML OUTPUT_DIR RENDER_GPU INFER_GPU [PORT]" >&2
    exit 2
fi

PRECISION="$1"
INFRA_ENGINE="$(readlink -f "$2")"
EGO_ENGINE="$(readlink -f "$3")"
ROUTE_XML="$(readlink -f "$4")"
OUTPUT_DIR="$5"
RENDER_GPU="$6"
INFER_GPU="$7"
PORT="${8:-40000}"
MAX_FRAMES="${UNIV2X_CARLA_MAX_FRAMES:-1200}"
INFERENCE_INTERVAL="${UNIV2X_CARLA_INFERENCE_INTERVAL:-10}"
TRAFFIC_VEHICLES="${UNIV2X_CARLA_TRAFFIC_VEHICLES:-10}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CARLA_ROOT="${CARLA_ROOT:-/home/lixingfeng/UniAD_examine/Carla/carla-0.9.10.1}"
CARLA_ENV="${CARLA_ENV:-/home/lixingfeng/UniAD_examine/Carla/.conda/carla0910_route}"
TRT_ENV="${TRT_ENV:-/home/lixingfeng/anaconda3/envs/univ2x_trt_runtime}"
TRT_ROOT="${TRT_ROOT:-/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118}"
PLUGIN="${UNIV2X_PLUGIN:-/home/lixingfeng/UniAD_examine/UniV2X/deploy_int8/plugins_msda_fp32accum/build_univ2x_fp32accum/lib_uniad_plugins_trt10.9_x86_cu118.so}"
TEMPLATE_DIR="${UNIV2X_TEMPLATE_DIR:-/home/lixingfeng/UniAD_examine/UniV2X/deploy_int8/artifacts/export_inputs/frame0}"

mkdir -p "${OUTPUT_DIR}/requests"
SOCKET_PATH="/tmp/univ2x_${PORT}_$$_4090.sock"
SERVICE_JSON="${OUTPUT_DIR}/inference_service.json"
RESULT_JSON="${OUTPUT_DIR}/closed_loop_metrics.json"

for path in "$INFRA_ENGINE" "$EGO_ENGINE" "$ROUTE_XML" "$PLUGIN" \
    "$TEMPLATE_DIR/ego.npz" "$TEMPLATE_DIR/infrastructure.npz" \
    "$CARLA_ROOT/CarlaUE4.sh" "$CARLA_ENV/bin/python" "$TRT_ENV/bin/python"; do
    if [[ ! -e "$path" ]]; then
        echo "required path not found: $path" >&2
        exit 1
    fi
done

CARLA_PID=""
SERVICE_PID=""
cleanup() {
    if [[ -n "$SERVICE_PID" ]]; then
        kill "$SERVICE_PID" 2>/dev/null || true
        wait "$SERVICE_PID" 2>/dev/null || true
    fi
    if [[ -n "$CARLA_PID" ]]; then
        kill -- "-$CARLA_PID" 2>/dev/null || true
        kill "$CARLA_PID" 2>/dev/null || true
        wait "$CARLA_PID" 2>/dev/null || true
    fi
    [[ -S "$SOCKET_PATH" ]] && unlink "$SOCKET_PATH" || true
}
trap cleanup EXIT

env CUDA_VISIBLE_DEVICES="$RENDER_GPU" setsid \
    "$CARLA_ROOT/CarlaUE4.sh" --world-port="$PORT" -prefer-nvidia -opengl -RenderOffScreen \
    >"${OUTPUT_DIR}/carla_server.log" 2>&1 &
CARLA_PID=$!

for _ in $(seq 1 120); do
    if "$CARLA_ENV/bin/python" - "$PORT" <<'PY' 2>/dev/null
import socket
import sys
s = socket.socket()
s.settimeout(1.0)
try:
    s.connect(("127.0.0.1", int(sys.argv[1])))
    s.close()
except Exception:
    sys.exit(1)
PY
    then
        break
    fi
    sleep 1
done

env CUDA_VISIBLE_DEVICES="$INFER_GPU" \
    PYTHONNOUSERSITE=1 \
    LD_LIBRARY_PATH="$TRT_ROOT/lib:$TRT_ENV/lib:$TRT_ENV/lib64" \
    "$TRT_ENV/bin/python" "$ROOT/deploy_int8/carla/univ2x_inference_service.py" \
    "$INFRA_ENGINE" "$EGO_ENGINE" \
    --plugin "$PLUGIN" \
    --template-input-dir "$TEMPLATE_DIR" \
    --socket "$SOCKET_PATH" \
    --output "$SERVICE_JSON" \
    --precision "$PRECISION" \
    --fixed-track-count 960 \
    --fixed-coop-count 64 \
    >"${OUTPUT_DIR}/inference_service.log" 2>&1 &
SERVICE_PID=$!

for _ in $(seq 1 180); do
    [[ -S "$SOCKET_PATH" ]] && break
    if ! kill -0 "$SERVICE_PID" 2>/dev/null; then
        cat "${OUTPUT_DIR}/inference_service.log" >&2
        exit 1
    fi
    sleep 1
done
if [[ ! -S "$SOCKET_PATH" ]]; then
    echo "inference service did not create $SOCKET_PATH" >&2
    exit 1
fi

CARLA_EGG="$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"
env PYTHONNOUSERSITE=1 PYTHONPATH="$CARLA_EGG:$CARLA_ROOT/PythonAPI/carla" \
    "$CARLA_ENV/bin/python" "$ROOT/deploy_int8/carla/run_univ2x_closed_loop.py" \
    --port "$PORT" \
    --socket "$SOCKET_PATH" \
    --route "$ROUTE_XML" \
    --output "$RESULT_JSON" \
    --request-dir "${OUTPUT_DIR}/requests" \
    --max-frames "$MAX_FRAMES" \
    --inference-interval "$INFERENCE_INTERVAL" \
    --traffic-vehicles "$TRAFFIC_VEHICLES" \
    >"${OUTPUT_DIR}/closed_loop_driver.log" 2>&1

wait "$SERVICE_PID"
SERVICE_PID=""
echo "closed-loop metrics: $RESULT_JSON"
