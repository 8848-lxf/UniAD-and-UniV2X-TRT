#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
    echo "usage: $0 PRECISION ENGINE FIXED_TRACK_COUNT ROUTE_XML OUTPUT_DIR RENDER_GPU INFER_GPU [PORT]" >&2
    exit 2
fi

PRECISION="$1"
ENGINE="$(readlink -f "$2")"
FIXED_TRACK_COUNT="$3"
ROUTE_XML="$(readlink -f "$4")"
OUTPUT_DIR="$5"
RENDER_GPU="$6"
INFER_GPU="$7"
PORT="${8:-40100}"
MAX_FRAMES="${UNIAD_CARLA_MAX_FRAMES:-1200}"
INFERENCE_INTERVAL="${UNIAD_CARLA_INFERENCE_INTERVAL:-10}"
TRAFFIC_VEHICLES="${UNIAD_CARLA_TRAFFIC_VEHICLES:-0}"
ROUTE_LOOKAHEAD_M="${UNIAD_CARLA_ROUTE_LOOKAHEAD_M:-8.0}"
MODEL_HEADING_WEIGHT="${UNIAD_CARLA_MODEL_HEADING_WEIGHT:-0.0}"
BLOCKED_FRAMES="${UNIAD_CARLA_BLOCKED_FRAMES:-400}"
SPEED_EMA_ALPHA="${UNIAD_CARLA_SPEED_EMA_ALPHA:-0.35}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CARLA_ROOT="${CARLA_ROOT:-/home/lixingfeng/UniAD_examine/Carla/carla-0.9.10.1}"
CARLA_ENV="${CARLA_ENV:-/home/lixingfeng/UniAD_examine/Carla/.conda/carla0910_route}"
TRT_ENV="${UNIAD_TRT_ENV:-/home/lixingfeng/UniAD_examine/.conda/univ2x_trt107_runtime}"
TRT_ROOT="${UNIAD_TRT_ROOT:-/data/lxf/uniad_deployment_outputs/toolchains/TensorRT-10.7.0-cu118-exact}"
TRT_PY_LIB="${UNIAD_TRT_PY_LIB:-$TRT_ENV/lib/python3.8/site-packages/tensorrt_libs}"
PLUGIN="${UNIAD_PLUGIN:-/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/trt107/runtime_build_trt107_exact/libuniad_plugin.so}"
CUDNN_LIB="${UNIAD_CUDNN_LIB:-/home/lixingfeng/anaconda3/envs/modelopt_uniad_dl4agx/lib/python3.10/site-packages/nvidia/cudnn/lib}"

mkdir -p "${OUTPUT_DIR}/requests"
SOCKET_PATH="/tmp/uniad_${PORT}_$$_4090.sock"
SERVICE_JSON="${OUTPUT_DIR}/inference_service.json"
RESULT_JSON="${OUTPUT_DIR}/closed_loop_metrics.json"

for path in "$ENGINE" "$ROUTE_XML" "$PLUGIN" "$CARLA_ROOT/CarlaUE4.sh" \
    "$CARLA_ENV/bin/python" "$TRT_ENV/bin/python" "$TRT_ROOT/lib/libnvinfer.so" \
    "$TRT_PY_LIB/libnvinfer.so.10"; do
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
    LD_LIBRARY_PATH="$TRT_ROOT/lib:$TRT_PY_LIB:$TRT_ENV/lib:$TRT_ENV/lib64:$CUDNN_LIB" \
    "$TRT_ENV/bin/python" "$ROOT/UniAD/deploy/carla/uniad_inference_service.py" \
    "$ENGINE" \
    --plugin "$PLUGIN" \
    --socket "$SOCKET_PATH" \
    --output "$SERVICE_JSON" \
    --precision "$PRECISION" \
    --fixed-track-count "$FIXED_TRACK_COUNT" \
    --context-cache-size 1 \
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
    "$CARLA_ENV/bin/python" "$ROOT/UniAD/deploy/carla/run_uniad_closed_loop.py" \
    --port "$PORT" \
    --socket "$SOCKET_PATH" \
    --route "$ROUTE_XML" \
    --output "$RESULT_JSON" \
    --request-dir "${OUTPUT_DIR}/requests" \
    --max-frames "$MAX_FRAMES" \
    --inference-interval "$INFERENCE_INTERVAL" \
    --traffic-vehicles "$TRAFFIC_VEHICLES" \
    --route-lookahead-m "$ROUTE_LOOKAHEAD_M" \
    --model-heading-weight "$MODEL_HEADING_WEIGHT" \
    --blocked-frames "$BLOCKED_FRAMES" \
    --speed-ema-alpha "$SPEED_EMA_ALPHA" \
    >"${OUTPUT_DIR}/closed_loop_driver.log" 2>&1

wait "$SERVICE_PID"
SERVICE_PID=""
echo "closed-loop metrics: $RESULT_JSON"
