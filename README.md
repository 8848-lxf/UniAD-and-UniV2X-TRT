# UniAD and UniV2X TensorRT Deployment

This repository tracks the RTX 4090 reproduction work for UniAD and UniV2X ONNX export, explicit-QDQ quantization, TensorRT engine construction, full-validation evaluation, and CARLA integration.

The canonical local deployment root is now `/home/lixingfeng/UniAD_examine/DL4AGX`:

- `DL4AGX/AV-Solutions/uniad-trt` is the UniAD deployment chain.
- `DL4AGX/AV-Solutions/vad-trt` is the VAD deployment chain.
- `DL4AGX/AV-Solutions/univ2x-trt` is the UniV2X deployment chain.
- `DL4AGX/AV-Solutions/docs` contains the three model handoff documents.

The checked-in tree remains the lightweight remote handoff checkout. The local `UniV2X/deploy_int8` path is retained as a compatibility symlink to the canonical UniV2X directory; generated artifacts stay outside source-control.

## Repository layout

- `UniAD/repro`: UniAD export, calibration, quantization, engine-build, and evaluation scripts plus the deployment-side model changes.
- `UniAD/runtime`: TensorRT enqueueV3 C++/CUDA runtime source.
- `UniAD/evidence`: small machine-readable reports retained for audit.
- `UniV2X/deploy_int8`: two-agent export, calibration, quantization, TensorRT runtime, plugin, evaluation, and CARLA adapter source.
- `UniV2X/projects`: UniV2X configs and model code used by the deployment flow.
- `UniV2X/evidence`: PyTorch/FP16 validation summaries and numerical-stability reports.
- `docs/codex_handoffs`: timestamped continuation notes for each model.

## Artifact policy

Model weights, ONNX graphs, TensorRT engines, timing/calibration caches, raw datasets, large per-frame tensors, compiled libraries, and build trees are intentionally excluded. The handoff documents record their expected local paths and lineage without committing the files.

Active development branch: `feature/uniad-univ2x-trt-4090`.

## Current status

- [UniAD handoff](docs/codex_handoffs/UniAD-4090.md)
- [UniV2X handoff](docs/codex_handoffs/UniV2X-4090.md)
