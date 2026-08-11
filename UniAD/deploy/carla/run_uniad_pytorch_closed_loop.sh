#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 7 ]]; then
    echo "usage: $0 CONFIG CHECKPOINT FIXED_TRACK_COUNT ROUTE_XML OUTPUT_DIR RENDER_GPU INFER_GPU [PORT]" >&2
    exit 2
fi

CONFIG="$(readlink -f "$1")"
CHECKPOINT="$(readlink -f "$2")"
FIXED_TRACK_COUNT="$3"
ROUTE_XML="$(readlink -f "$4")"
OUTPUT_DIR="$5"
RENDER_GPU="$6"
INFER_GPU="$7"
PORT="${8:-40100}"
MAX_FRAMES="${UNIAD_CARLA_MAX_FRAMES:-1200}"
INFERENCE_INTERVAL="${UNIAD_CARLA_INFERENCE_INTERVAL:-10}"
TRAFFIC_VEHICLES="${UNIAD_CARLA_TRAFFIC_VEHICLES:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CARLA_ROOT="${CARLA_ROOT:-/home/lixingfeng/UniAD_examine/Carla/carla-0.9.10.1}"
CARLA_ENV="${CARLA_ENV:-/home/lixingfeng/UniAD_examine/Carla/.conda/carla0910}"
PYTORCH_ENV="${UNIAD_PYTORCH_ENV:-/home/lixingfeng/anaconda3/envs/torch112}"
DEPLOY_ROOT="${UNIAD_DEPLOY_ROOT:-/home/lixingfeng/UniAD_examine/DL4AGX/AV-Solutions/uniad-trt/repro_20260806/UniAD_deploy}"

mkdir -p "${OUTPUT_DIR}/requests"
SOCKET_PATH="/tmp/uniad_pytorch_${PORT}_$$_4090.sock"
SERVICE_JSON="${OUTPUT_DIR}/inference_service.json"
RESULT_JSON="${OUTPUT_DIR}/closed_loop_metrics.json"

for path in "$CONFIG" "$CHECKPOINT" "$ROUTE_XML" "$CARLA_ROOT/CarlaUE4.sh" \
    "$CARLA_ENV/bin/python" "$PYTORCH_ENV/bin/python"; do
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

env CUDA_VISIBLE_DEVICES="$INFER_GPU" PYTHONNOUSERSITE=1 \
    UNIAD_DEPLOY_ROOT="$DEPLOY_ROOT" \
    PYTHONPATH="$DEPLOY_ROOT" \
    LD_LIBRARY_PATH="$PYTORCH_ENV/lib:$PYTORCH_ENV/lib64" \
    "$PYTORCH_ENV/bin/python" "$ROOT/UniAD/deploy/carla/uniad_pytorch_inference_service.py" \
    "$CONFIG" "$CHECKPOINT" --socket "$SOCKET_PATH" --output "$SERVICE_JSON" \
    --fixed-track-count "$FIXED_TRACK_COUNT" \
    >"${OUTPUT_DIR}/inference_service.log" 2>&1 &
SERVICE_PID=$!

for _ in $(seq 1 300); do
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
    --port "$PORT" --socket "$SOCKET_PATH" --route "$ROUTE_XML" \
    --output "$RESULT_JSON" --request-dir "${OUTPUT_DIR}/requests" \
    --max-frames "$MAX_FRAMES" --inference-interval "$INFERENCE_INTERVAL" \
    --traffic-vehicles "$TRAFFIC_VEHICLES" \
    >"${OUTPUT_DIR}/closed_loop_driver.log" 2>&1

wait "$SERVICE_PID"
SERVICE_PID=""
echo "closed-loop metrics: $RESULT_JSON"
