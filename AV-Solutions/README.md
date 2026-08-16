# Unified deployment handoff

The maintained deployment chains in this checkout are:

- `uniad-trt`: UniAD-tiny/base export, TensorRT runtime, calibration, and evaluation.
- `vad-trt`: VAD-Tiny first-frame/previous-frame export and TensorRT benchmark path.
- `univ2x-trt`: UniV2X infrastructure and ego-agent export, calibration, engines, and CARLA adapters.
- `docs/`: the three PDT-timestamped development handoff documents.

Large generated outputs are intentionally local-only. Use the links and paths recorded in the handoff documents instead of adding engines, ONNX files, checkpoints, datasets, or build directories to Git.
