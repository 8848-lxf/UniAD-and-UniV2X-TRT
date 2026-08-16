# UniV2X TensorRT deployment workspace

Canonical local deployment root:

`/home/lixingfeng/UniAD_examine/DL4AGX/AV-Solutions/univ2x-trt`

The former `/home/lixingfeng/UniAD_examine/UniV2X/deploy_int8` path is a compatibility symlink to this directory. Large generated artifacts remain outside the source tree and are exposed locally through the `artifacts` symlink; engines, ONNX files, calibration blobs, and checkpoints are excluded from source-control handoffs.

This directory contains an isolated deployment port for the cooperative
UniV2X stage-2 checkpoint. Source datasets and checkpoints are consumed
read-only from their existing locations. Generated ONNX, calibration, engine,
evaluation, and latency artifacts live under `artifacts/`.

The current artifact symlink resolves to
`/data/lxf/univ2x_deployment_outputs/semantic_parity_20260810`; generated
files therefore stay on the data filesystem rather than consuming home
storage.

The deployment is split into infrastructure and ego-agent graphs because the
original PyTorch model executes the infrastructure agent first and feeds its
cooperative features into the ego agent. Framework and end-to-end latency are
reported separately.
