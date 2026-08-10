# UniAD-tiny current-checkpoint and official-dummy comparison

Recorded: 2026-08-08 (Asia/Shanghai)

## Executive result

- Training had 3 complete stage-1 epochs when this audit started. It was resumed from `epoch_3.pth` on GPUs 0-5; `epoch_4.pth` is now complete and stage-1 epoch 5 is running. GPUs 6 and 7 are excluded from training and are free for evaluation.
- The earlier engineering smoke did use an E2E checkpoint, but it was **UniAD-base E2E**, not a trained UniAD-tiny checkpoint. Converting it to tiny left 65 shape-conflicting tensors reinitialized and 446 source tensors unused, so its planning accuracy is not an official-quality result.
- NVIDIA's distributed `uniad_tiny_dummy.onnx` is explicitly random-weight and cannot reproduce accuracy. In this run its FP32 and FP16 planning outputs were non-finite; the sanitized INT8 graph returned a finite but meaningless `394.8122 m` average L2.
- The available tiny checkpoint tested here is stage-1 epoch 3, not a finished E2E model and not an empirically selected "best" checkpoint. Loaded into the stage-2 config, the motion/occupancy/planning heads are random. Its full validation result (`L2=11.9146 m`, box collision `15.6698%`) therefore does not measure a trained UniAD-tiny planner.
- On the RTX 4090, the official dummy graph follows `FP32 > FP16` but **does not** follow the official `FP16 > INT8(EQ)+FP16` speed trend. The controlled `trtexec` GPU-compute medians are `15.1071 / 13.0032 / 18.0700 ms` for FP32/FP16/INT8 respectively. The full 6018-frame app confirms INT8 mean model latency is 9.2% slower than FP16.
- A valid accuracy and final speed-trend comparison requires the documented stage-1 6-epoch plus stage-2 20-epoch tiny E2E checkpoint, export of that same checkpoint, and representative entropy calibration. Those prerequisites do not yet exist in the current training run.

## Training and GPU status

| Item | Status |
|---|---|
| Complete at audit start | stage-1 epochs 1-3 |
| Complete now | stage-1 epochs 1-4 |
| Active training | stage-1 epoch 5, GPUs 0-5 |
| Reserved GPUs | GPUs 6 and 7, no compute processes after evaluation cleanup |
| Resume checkpoint | `artifacts/stage1/epoch_3.pth`, SHA256 `89234539dd155796a4649442bba9a1c2d2dda03e197ee6ce9120cd71b83faa80` |
| Latest complete checkpoint | `artifacts/stage1/epoch_4.pth`, SHA256 `6295687f9dbec25488f75b86ac79c4a46a341be196dd0fd4d3c37e8491a9d99a` |

The original run used 8 GPUs. To reserve two GPUs, the partial epoch 4 was discarded and training resumed from the validated epoch-3 checkpoint with 6 GPUs. This changes global batch size from 8 to 6 while retaining the configured learning rate, so the resulting optimization trajectory is not bitwise or training-protocol equivalent to NVIDIA's reference.

## Weight lineage

| Purpose | Weight | SHA256 | Interpretation |
|---|---|---|---|
| Earlier deployment smoke | `/home/lixingfeng/data/ckpts/uniad_base_e2e.pth` | `0ad0c2f5dc9788a41c313305779ea49346aeb742d1f6bb5ad25c46f9beffc990` | E2E, but UniAD-base rather than tiny |
| Formal tiny initialization | `/data/lxf/uniad_ckpts/bevformer_tiny_epoch_24.pth` | `7305046dbaa4fe8b1fa6d6acb9e0e3d605a70a3c473f763e936103428d2b2f12` | BEVFormer-tiny initialization, not E2E |
| Tested current tiny | `artifacts/stage1/epoch_3.pth` | `89234539dd155796a4649442bba9a1c2d2dda03e197ee6ce9120cd71b83faa80` | Stage-1 track/map checkpoint; no trained motion/occ/planning heads |
| NVIDIA sample ONNX | `artifacts/official_dummy/onnx/uniad_tiny_dummy.onnx` | `3f849b4a45179e31a1537b8bde8a83d643a414b22fa644a775f68c989f700648` | Random weights by NVIDIA's documentation |

The base-to-tiny smoke transferred 1946 exact-shape tensors out of 2011 tiny target tensors (97.37% of target parameter elements), but 65 target tensors had incompatible shapes and 446 base tensors were unexpected. BEV size, feature levels, deformable-attention sampling tensors, and related heads differ between base and tiny. High planning error from this hybrid is expected.

## Official reference

NVIDIA reports the following on XAVIER using TensorRT 10.7. These are reference values, not measurements from this host.

| Framework | Precision | DL model latency (ms) | FPS | avg. L2 | avg. Col | planning MSE |
|---|---:|---:|---:|---:|---:|---:|
| PyTorch 1.12 | FP32 | 843.5172 | 1.18 | 0.9986 | 0.27 | 0 |
| TensorRT 10.7 | FP32 | 64.0469 | 15.61 | 0.9986 | 0.27 | 9.2417e-07 |
| TensorRT 10.7 | FP16 | 49.7559 | 20.10 | 1.0021 | 0.26 | 0.0458 |
| TensorRT 10.7 | INT8(EQ)+FP16 | 39.3125 | 25.44 | 1.0029 | 0.27 | 0.0502 |

Sources:

- https://github.com/NVIDIA/DL4AGX/blob/master/AV-Solutions/uniad-trt/documents/train_export.md
- https://github.com/NVIDIA/DL4AGX/blob/master/AV-Solutions/uniad-trt/documents/explicit_quantization.md

## Current stage-1 tiny checkpoint evaluation

The test used stage-1 `epoch_3.pth` with the stage-2 tiny E2E config over all 6019 validation samples on GPU 7. The main forward loop and planning metrics completed. The subsequent nuScenes detection evaluator was stopped after it began processing about 1.36 million boxes from random downstream heads; this released GPU 7 and does not change the already-written planning or latency results.

| Model | Frames | Model forward mean / p50 / p95 (ms) | E2E mean / p50 / p95 (ms) | avg. L2 (m) | avg. box Col | avg. point Col |
|---|---:|---:|---:|---:|---:|---:|
| Stage-1 epoch 3 loaded as E2E | 6019 | 1915.220 / 1998.788 / 2300.983 | 1967.590 / 2050.234 / 2360.196 | 11.9146 | 15.6698% | 5.6848% |

The checkpoint was the latest complete one when the full evaluation started. Epoch 4 became available only after the evaluation was mostly complete. Neither checkpoint can be called the "best tiny E2E" model: stage-1 has no trained planning head, and no per-epoch E2E planning validation exists yet. Running epoch 4 through the same stage-2 evaluator would chiefly measure another random planning-head initialization, not meaningful planning progress.

## Official random ONNX test

### Accuracy

| Engine | Frames | avg. L2 | avg. box Col | Validity |
|---|---:|---:|---:|---|
| FP32 | 6018 | NaN | 3.1406% | All trajectories non-finite; invalid |
| FP16 | 6018 | NaN | 3.1406% | All trajectories non-finite; invalid |
| INT8(EQ)+FP16 | 6018 | 394.8122 m | 3.1406% | Finite only after quantization/clipping; physically meaningless |

The collision number cannot rescue these results: a random or non-finite trajectory is not an accuracy baseline. NVIDIA documents this ONNX as a legal-distribution convenience with random weights, not as the model that produced its accuracy table.

### Full application latency

These results contain 6018 frames and exclude 10 warm-up frames. `model` is a CUDA event around `enqueueV3`; `call` is synchronized H2D + inference + dynamic-output handling + D2H; `E2E` additionally includes metadata, six-JPEG decode, GPU preprocessing, temporal update, and output decoding.

| Engine | Model mean / p50 / p95 (ms) | Call mean / p50 / p95 (ms) | E2E mean / p50 / p95 (ms) | Model FPS | E2E FPS |
|---|---:|---:|---:|---:|---:|
| FP32 | 19.312 / 18.267 / 27.587 | 23.718 / 22.427 / 33.862 | 140.637 / 124.958 / 219.110 | 51.78 | 7.11 |
| FP16 | 16.503 / 15.906 / 20.334 | 20.614 / 19.990 / 25.312 | 115.097 / 112.715 / 138.227 | 60.59 | 8.69 |
| INT8(EQ)+FP16 | 18.024 / 17.217 / 23.886 | 23.246 / 22.258 / 30.788 | 120.090 / 117.425 / 144.797 | 55.48 | 8.33 |

FP32 and FP16 use temporal state. The random INT8 graph emits 1801 tracks, exceeding the engine's temporal input limit of 1150, so its app result uses an explicit independent-frame benchmark mode. Consequently, use the controlled `trtexec` table below for the cleanest precision trend comparison; the app result is corroborating evidence, not a perfectly identical stateful workload.

### Controlled TensorRT latency

This follows the official style more closely: fixed input shapes, 200 ms warm-up, 100 timed iterations, and GPU Compute Time from `trtexec` on GPU 6.

| Engine | GPU compute mean (ms) | p50 (ms) | p95 (ms) | Trend vs previous |
|---|---:|---:|---:|---|
| FP32 | 15.2791 | 15.1071 | 17.9825 | baseline |
| FP16 | 14.8625 | 13.0032 | 27.4207 | p50 1.162x faster than FP32 |
| INT8(EQ)+FP16 | 20.8950 | 18.0700 | 34.0367 | p50 1.390x slower than FP16 |

Verdict: the observed speed ordering is `FP16 < FP32 < INT8` by p50 latency, whereas NVIDIA reports `INT8 < FP16 < FP32`. Therefore the current INT8 trend does **not** reproduce the official trend.

## Why INT8 is not faster here

1. NVIDIA's recommended entropy calibration fails on the random dummy graph because random activations overflow its histogram path. This run used max calibration with one real input sample and then removed exactly three non-finite QDQ pairs. It is an engineering-runnable fallback, not the official calibration result.
2. The final sanitized graph passes ONNX and deployment-contract validation and contains 523 `DequantizeLinear` nodes plus 672 INT8 initializers, with no MatMul INT8 weights. Engine inspection confirms real INT8 execution, but only 443 of 2582 engine layers expose INT8 tensors; remaining FP16/plugin work and reformat boundaries can dominate.
3. UniAD has many small operators, dynamic-output operations, plugins, and memory traffic. Partial quantization can add QDQ/reformat overhead without enough Tensor Core work to amortize it.
4. The random graph produces pathological dynamic track counts. This alters runtime behavior and is not representative of a trained E2E model.
5. The hardware differs from NVIDIA's XAVIER reference. This can change absolute latency and kernel selection, but by itself does not validate the reversed trend.

The current INT8 engine is genuinely explicit-quantized, rather than an FP16 engine carrying an INT8 label. The issue is ineffective/partial quantization for this random graph and fallback calibration, not simply an engine naming error.

## Why earlier avg. L2 and avg. Col were high

The earlier 100-frame hybrid smoke measured `L2=6.1967 m` and box collision `0.8333%` for FP32. Its gap from the official `0.9986 / 0.27` comes from several confounders:

1. Base E2E weights were forced into a smaller tiny architecture with known shape conflicts and missing target tensors.
2. It used only 100 frames rather than the full 6018/6019-frame validation set; collision is especially noisy at that scale.
3. INT8 calibration used one sample, far below a representative calibration set.
4. All eight GPUs were training concurrently, creating multi-second scheduling/host-contention stalls. The sub-second samples had model p50 values of about 28.283/18.785/20.363 ms for FP32/FP16/INT8, which already showed FP16 faster than INT8.
5. Official `avg. Col` is reported in percentage points. Mixing fraction and percent representations can create an additional 100x interpretation error.

The full stage-1 checkpoint result is worse (`11.9146 m`, `15.6698%`) because its planning head is entirely untrained; it is not evidence that continued tiny training degraded a trained planner.

## Reproduction gate

Accuracy comparison against the NVIDIA table is currently **not passed**. The random ONNX and stage-1 checkpoint are both structurally unsuitable for that claim. The next valid gate is:

1. Complete stage-1 epoch 6 and stage-2 epoch 20 according to the tiny training recipe.
2. Evaluate candidate stage-2 checkpoints in PyTorch over the full validation set and select using real E2E planning metrics.
3. Export that exact trained tiny E2E checkpoint to ONNX.
4. Run representative entropy calibration without non-finite scale repair, rebuild FP32/FP16/INT8 engines, and verify graph/engine precision lineage.
5. Repeat full-set accuracy and isolated-GPU `trtexec` plus application E2E latency. Only then compare precision drift and speed ordering with NVIDIA's table.

## Primary artifacts

- Current tiny PyTorch metrics: `artifacts/stage1_epoch3_e2e_baseline/evaluation/pytorch_fp32/latency_metrics.json`
- Current tiny planning predictions: `artifacts/stage1_epoch3_e2e_baseline/evaluation/pytorch_fp32/planning_predictions.csv`
- Official dummy FP32/FP16/INT8 app metrics: `artifacts/official_dummy/evaluation/tensorrt_*/latency_metrics.json`
- Official dummy planning metrics: `artifacts/official_dummy/evaluation/tensorrt_*/planning_metrics.json`
- Controlled TensorRT logs: `artifacts/official_dummy/evaluation/trtexec/*.log`
- Quantization audit: `artifacts/official_dummy/onnx/quantization_inspection.json`
- Three-QDQ repair manifest: `artifacts/official_dummy/onnx/nonfinite_qdq_sanitization.json`
- Engine layer inspections: `artifacts/official_dummy/engine_inspection/*_layers.json`
- Earlier base-to-tiny conversion audit: `artifacts/smoke_base_hybrid/checkpoints/conversion_audit.json`
