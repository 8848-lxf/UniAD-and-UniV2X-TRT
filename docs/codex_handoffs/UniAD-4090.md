# UniAD TensorRT/INT8 4090 Handoff

Timeline timestamps use the active process timezone (PDT, UTC-07:00).

Status is fail-closed: a smoke test is not counted as full validation, and the official random-weight ONNX results are not treated as model-accuracy evidence.

## Local lineage

- Working root: `/home/lixingfeng/UniAD_examine/DL4AGX/AV-Solutions/uniad-trt/repro_20260806`
- Original checkpoint: `/home/lixingfeng/data/ckpts/uniad_base_e2e.pth`
- Trained tiny checkpoint: `repro_20260806/artifacts/stage2/epoch_20.pth`, SHA-256 `88289ecdf2a7576a4c6bd1c0879d15ffba5b93fc22221872fef6196638d91484`
- Dataset: `/data/uniad_data` through local read-only links
- PyTorch environment: existing isolated UniAD/PyTorch 1.12 environment
- Deployment environment: `modelopt_uniad_dl4agx`
- CUDA compiler: Conda environment toolchain only
- TensorRT: `/home/lixingfeng/UniAD_examine/HEAL/prune_model/TensorRT-10.9_x86_cu118`
- GPU used for current full engine evaluation: RTX 4090, logical GPU 6

## Completion matrix

| Item | State | Evidence/notes |
| --- | --- | --- |
| NVIDIA tiny/random ONNX reference flow | Completed as a deployment smoke | All three precisions ran; random weights make its planning accuracy non-physical and unsuitable for accuracy reproduction. |
| Trained UniAD-tiny stage 1/2 | Completed | Stage 1 epoch 6 and stage 2 epoch 20 follow `documents/train_export.md`; both training and deployment `ckpts` entries resolve to one registry. |
| Trained UniAD-tiny PyTorch full validation | Completed | 6019 frames; detection, tracking, map, occupancy, planning, and latency are recorded below. |
| Trained UniAD-tiny ONNX/calibration/engines | Completed | Real epoch-20 checkpoint exported with the tutorial TRT config; 64 training samples and FP32/FP16/INT8(EQ)+FP16 engines are available locally. |
| Trained UniAD-tiny temporal TensorRT evaluation | Completed | Corrected initial padding, OpenCV-compatible resize, scene reset, exact TensorRT 10.7 runtime, and fixed 1300/1600 contracts completed 6018 frames for FP32/FP16/INT8. |
| UniAD-base config and checkpoint adaptation | Completed | Uses the base graph, base input metadata, and dynamic temporal-track profile; the checkpoint is not inserted into the tiny graph. |
| Base FP32 ONNX export | Completed | Local artifact excluded from Git. |
| Base explicit-QDQ INT8 graph | Completed | ONNX check passes with expected TRT plugin-domain handling; MatMul weights/activations are excluded from INT8. |
| Base FP32/FP16/INT8 engine build | Completed | Local `.engine` files excluded from Git. |
| Base engine finite-output smoke | Completed | `UniAD/evidence/uniad_base_e2e/base_engine_smoke_sample0.json`. |
| Base PyTorch full 6018-frame evaluation | Completed | Original checkpoint baseline recorded below. |
| Base TensorRT FP32 dynamic full 6018-frame evaluation | Completed, not accepted as final runtime | Accuracy and latency are recorded below. Forty-two shape-update spikes above 1 s inflated the mean; fixed-shape rerun supersedes its timing. |
| Base TensorRT FP32 fixed-1150 full evaluation | Completed | All 6018 frames completed with finite metrics and no second-scale latency spikes. |
| Base TensorRT FP16 fixed-1150 full evaluation | Completed | All 6018 frames completed with finite planning and latency summaries. |
| Base TensorRT INT8 fixed-1150 full evaluation | Completed, preliminary calibration | All 6018 frames completed; its evidence path explicitly records the current 8-sample training calibration. |
| CARLA closed-loop evaluation | Completed as a custom-route diagnostic | PyTorch and all three TensorRT precisions run for 1200 Town03 frames with identical sensors/controller. This is not an official Bench2Drive score because the complete official assets/protocol are unavailable. |

## Verified results

### Original PyTorch checkpoint, full validation

| Metric | Value |
| --- | ---: |
| mAP | 0.380395 |
| NDS | 0.498536 |
| AMOTA | 0.361174 |
| AMOTP | 1.331018 |
| planning avg. L2 | 0.913059 m |
| planning avg. box collision | 0.066456% |
| forward mean / p50 / p95 | 517.867 / 470.277 / 769.141 ms |
| end-to-end mean / p50 / p95 | 586.327 / 545.333 / 840.863 ms |

### Base engine smoke, one sample

| Precision | Forward mean | Forward p50 | Status |
| --- | ---: | ---: | --- |
| FP32 | 192.667 ms | 187.682 ms | finite |
| FP16 | 113.91 ms | recorded in evidence | finite |
| INT8(EQ)+FP16 | 104.18 ms | recorded in evidence | finite |

The smoke latency trend is `FP32 > FP16 > INT8`, matching the NVIDIA example qualitatively. It is not a substitute for the full 6018-frame accuracy and latency comparison.

### Base TensorRT FP32, dynamic full validation

| Metric | Value |
| --- | ---: |
| planning avg. L2 | 2.559613 m |
| planning avg. point collision, corrected base bounds | 0.254791% |
| planning avg. box collision, corrected base bounds | 1.165947% |
| enqueue mean / p50 / p99 | 262.848 / 185.484 / 195.459 ms |
| inference-call mean / p50 / p99 | 292.432 / 214.441 / 242.205 ms |
| end-to-end mean / p50 / p99 | 447.977 / 366.110 / 474.084 ms |

The mean includes 42 enqueue spikes above 1 s, with maxima around 11.7 s. The original standalone evaluator also used tiny `50x50` collision bounds for this base `200x200` run; the corrected values above use the base `[-50, 50, 0.5]` grid. L2 is unaffected by that bounds bug.

### Base TensorRT FP32, fixed-1150 full validation

| Metric | Value |
| --- | ---: |
| planning avg. L2 | 2.559613 m |
| planning avg. point collision | 0.254791% |
| planning avg. box collision | 1.165947% |
| planning vs PyTorch output avg. L2 | 3.144457 m |
| enqueue mean / p50 / p99 | 185.945 / 185.443 / 191.793 ms |
| inference-call mean / p50 / p99 | 215.580 / 214.395 / 233.181 ms |
| end-to-end mean / p50 / p99 | 368.480 / 364.289 / 440.199 ms |

The fixed-input run preserved the dynamic run's planning output but removed all 42 second-scale shape-reconfiguration spikes. Maximum enqueue was `209.019 ms` and maximum end-to-end was `625.901 ms`. Evidence is retained under `UniAD/evidence/uniad_base_e2e/fp32_fixed1150_full6018`.

### Base TensorRT fixed-1150 precision comparison

| Metric | FP32 | FP16 | INT8(EQ)+FP16, calib8 |
| --- | ---: | ---: | ---: |
| planning avg. L2 | 2.559613 m | 2.408856 m | 2.380163 m |
| planning avg. point collision | 0.254791% | 0.210480% | 1.082863% |
| planning avg. box collision | 1.165947% | 1.024704% | 3.162734% |
| planning vs PyTorch output avg. L2 | 3.144457 m | 2.966040 m | 2.722219 m |
| enqueue mean / p50 / p99 | 185.945 / 185.443 / 191.793 ms | 108.196 / 107.601 / 115.428 ms | 97.302 / 96.794 / 103.337 ms |
| inference-call mean / p50 / p99 | 215.580 / 214.395 / 233.181 ms | 137.403 / 135.917 / 162.725 ms | 126.623 / 125.679 / 143.087 ms |
| end-to-end mean / p50 / p99 | 368.480 / 364.289 / 440.199 ms | 289.559 / 284.672 / 379.506 ms | 276.915 / 273.425 / 349.483 ms |

The full-validation enqueue trend is `FP32 > FP16 > INT8`, matching NVIDIA's qualitative example. Relative to FP32, FP16 is `1.72x` faster and INT8 is `1.91x` faster; INT8 is `1.11x` faster than FP16. Accuracy does not reproduce NVIDIA's tiny-model example: the base PyTorch checkpoint has planning L2 `0.913059 m`, but even the FP32 TensorRT export has `2.559613 m`. Because this delta already exists in FP32, it is an export/runtime fidelity issue rather than an INT8 calibration-only issue. INT8 additionally raises box collision to `3.162734%`, and the 8-sample calibration remains preliminary.

The current NVIDIA-style C++ runner reconstructs planning and decoded boxes but only emits a full planning evaluation. It does not reconstruct the Python dataset's complete detection/tracking/map result bundle, so no TensorRT detection/tracking score is claimed here.

### Trained UniAD-tiny epoch 20

The training and export chain follows `documents/train_export.md`: tiny stage 1 runs to epoch 6, stage 2 runs to epoch 20, and ONNX export invokes `tools/export_onnx.py` with `projects/configs/stage2_e2e/tiny_imgx0.25_e2e_trt_p.py`. This run uses the real trained checkpoint, not NVIDIA's legal-placeholder random ONNX.

Both local checkpoint entry points resolve to `repro_20260806/artifacts/checkpoints`:

- `UniAD/ckpts`
- `UniAD_train/ckpts`

The registry exposes `tiny_imgx0.25_e2e_ep20.pth` and the tutorial-compatible alias `tiny_imgx0.25_e2e.pth`, both resolving to the 805,231,233-byte epoch-20 checkpoint. Original files under `/data` remain read-only symlink targets.

| PyTorch full-validation metric | Value |
| --- | ---: |
| frames | 6019 |
| mAP / NDS | 0.154061 / 0.300379 |
| AMOTA / AMOTP | 0.080732 / 1.787528 |
| map drivable / lanes IoU | 0.716173 / 0.337398 |
| map divider / crossing / contour IoU | 0.253431 / 0.125746 / 0.337554 |
| occupancy IoU class 0 / 1 | 51.4 / 48.6 |
| planning avg. L2 | 0.826428 m |
| planning point / box collision | 0.013845% / 0.221521% |
| forward mean / p50 / p95 | 230.429 / 185.852 / 490.950 ms |
| end-to-end mean / p50 / p95 | 270.950 / 226.852 / 534.329 ms |

This retained baseline predates the native p99 patch, so p99 is not inferred from p95. Future baseline timing reruns emit p99 directly.

The exported real-weight graph has a 64-sample training calibration package at `trained_tiny_epoch20/calibration/calib_data_shape0_901.npz`. The INT8 graph passes explicit-QDQ validation with MatMul excluded and finite scales.

| `trtexec`, 100 iterations at track shape 901 | FP32 | FP16 | INT8(EQ)+FP16 |
| --- | ---: | ---: | ---: |
| GPU compute mean / p50 / p99 | 13.507 / 13.332 / 17.194 ms | 9.471 / 9.417 / 11.166 ms | 10.008 / 9.917 / 12.040 ms |
| overall mean / p50 / p99 | 14.675 / 14.502 / 18.362 ms | 10.764 / 10.709 / 12.521 ms | 11.185 / 11.098 / 13.250 ms |

The table above is retained as the original `trtexec` diagnostic. It used the old shape-901 standalone contract, where FP16 was faster than INT8 on this RTX 4090. The corrected temporal application supersedes it for end-to-end deployment evaluation.

### Corrected trained-tiny temporal deployment, full 6018 frames

The accepted rerun uses the exact TensorRT 10.7 runtime expected by the NVIDIA project, OpenCV-compatible half-pixel image resize, real `-10000` padding of the initial temporal tensors, fixed 1300/1600 capacities, and per-scene reset metadata. All 6018 frames completed with finite output.

| Metric | Deployment PyTorch | TensorRT FP32 | TensorRT FP16 | TensorRT INT8(EQ)+FP16 |
| --- | ---: | ---: | ---: | ---: |
| planning avg. L2 | 0.780738 m | 0.780420 m | 0.758038 m | 0.801484 m |
| planning point collision | 0.085854% | 0.085854% | 0.091392% | 0.085854% |
| planning box collision | 0.667442% | 0.667442% | 0.631439% | 0.808685% |
| planning MSE vs PyTorch | 0 | 0.003772 m | 0.119177 m | 0.274626 m |
| enqueue mean / p50 / p99 | n/a | 17.371 / 17.258 / 19.669 ms | 13.043 / 12.907 / 16.896 ms | 12.889 / 12.800 / 16.750 ms |
| synchronized forward mean / p50 / p99 | n/a | 21.141 / 20.984 / 23.884 ms | 17.027 / 16.859 / 23.287 ms | 16.949 / 16.788 / 23.272 ms |
| end-to-end mean / p50 / p99 | n/a | 98.306 / 97.042 / 112.049 ms | 93.600 / 92.367 / 107.090 ms | 93.304 / 91.885 / 108.210 ms |

The accepted enqueue/forward ordering is `FP32 > FP16 > INT8`, matching the NVIDIA example qualitatively. Relative to FP32 enqueue, FP16 is `1.33x` faster and INT8 is `1.35x` faster; INT8 is only `1.01x` faster than FP16 on RTX 4090.

This is same-checkpoint accuracy parity, not exact reproduction of NVIDIA's hidden/placeholder-checkpoint values. FP32 planning quality matches its PyTorch reference closely, but its trajectory MSE `0.003772 m` is above NVIDIA's `9.2417e-07`; FP16/INT8 MSE is also larger than the example. The official C++ graph does not reconstruct the complete Python detection/tracking/map/occupancy result bundle, so engine mAP, AMOTA, map IoU, and occupancy IoU are unavailable rather than reported as zero. The original epoch-20 PyTorch full-task scores remain in the preceding table.

### CARLA Town03 model-controlled closed-loop diagnostic

The original epoch-20 PyTorch checkpoint and all three TensorRT precisions use the same six synchronized RGB cameras, route XML, preprocessing, recurrent state, controller, no traffic, 1200 frames, and inference every 10 frames. The latency table excludes the first cold inference.

| Backend | Route progress | Collision callbacks / lane invasions | Forward mean / p50 / p99 | Service E2E mean / p50 / p99 |
| --- | ---: | ---: | ---: | ---: |
| PyTorch FP32 | 27.90% | 1101 / 18 | 94.232 / 91.734 / 143.972 ms | 261.587 / 260.725 / 306.752 ms |
| TensorRT FP32 | 27.89% | 1096 / 18 | 20.374 / 19.641 / 29.731 ms | 214.238 / 204.636 / 266.631 ms |
| TensorRT FP16 | 27.94% | 1093 / 18 | 17.869 / 16.838 / 29.899 ms | 197.037 / 192.933 / 245.430 ms |
| TensorRT INT8(EQ)+FP16 | 27.99% | 1092 / 18 | 18.299 / 15.954 / 37.416 ms | 194.969 / 191.444 / 264.580 ms |

All variants follow the same initial trajectory and then stall in repeated contact around 28% route progress. Callback counts are repeated contact events, not unique collisions. The integration is live and model-controlled, but the reused CARLA tree does not contain the complete official Bench2Drive assets/protocol; these values must not be reported as an official closed-loop driving score. FP16 has the lowest forward mean, while INT8 has the lowest p50 and service end-to-end mean; unlike the open-loop enqueue result, CARLA INT8 mean is slightly above FP16 because its p99 tail is larger.

## Official-patch and base-port audit

The base deployment was not exported from unmodified UniAD code. It uses the same patched deployment tree required by NVIDIA's tutorial:

- `uniad-torch1.12.patch` aligns UniAD with the Torch 1.12/MMCV/MMDetection/MMDetection3D API set used by the deployment environment.
- `uniad-onnx-export.patch` supplies tensor-only `UniADTRT`/`UniADTrackTRT` forwards, explicit temporal state tensors, vectorized tracker/memory updates, custom `atan2`, TopK-based selection, and TRT-compatible implementations for tracking, map, motion, occupancy, and planning heads.
- `mmdet3d.patch` and `bevformer_tensorrt.patch` add the borrowed data structures, coders, ONNX symbolics, and TensorRT plugin bridges needed by deformable attention and related BEVFormer operators.
- `plugins-trt10-support.patch` updates plugin interfaces and dimension handling for TensorRT 10. The nuscenes and tiny-training patches primarily provide environment/evaluator and tiny-training compatibility rather than INT8 arithmetic.

`projects/configs/stage2_e2e/base_e2e_trt_p.py` inherits the real base configuration and replaces the deployment-sensitive modules with `UniADTRT`, `BEVFormerTrackHeadTRTP`, `PerceptionTransformerUniADTRTP`, the TRTP attention/encoder/decoder stack, `PansegformerHeadTRTP`, `OccHeadTRTP`, `MotionHeadTRTP`, and `PlanningHeadSingleModeTRTP`. It therefore preserves the base backbone, `200x200` BEV, and base tensor shapes rather than inserting the checkpoint into the tiny graph.

The quantization/build recipe also follows the tutorial structurally: ModelOpt 0.29.0, TensorRT 10.9, ONNX Runtime GPU 1.21.0, the locally rebuilt plugin, global MatMul exclusion, DQ-only explicit QDQ, temporal profiles from 901 to 1150, and `trtexec --best`. The current base INT8 graph differs in calibration evidence: it has only an 8-sample training calibration and remains preliminary.

The unresolved problem is equivalence validation. NVIDIA's published numbers validate the patched tiny graph; the base port exercises a much larger image/BEV topology and shape-dependent branches for which no official reference is provided. We adapted shapes, profiles, grids, and plugin inputs, but did not establish a passing PyTorch -> patched PyTorch -> ONNX Runtime -> TensorRT per-layer and cross-frame parity gate before full evaluation. Since FP32 TensorRT already changes planning L2 from `0.913059 m` to `2.559613 m` and differs from the PyTorch trajectory by `3.144457 m`, the base engine is a completed runtime deployment but not an accuracy-faithful reproduction. Larger INT8 calibration cannot repair that FP32 export/runtime delta.

## Data loading

The UniAD Python base and tiny evaluation configurations already set `workers_per_gpu=8`. The reported full TensorRT planning evaluations use the C++ application, which reads metadata and six JPEGs directly and has no Python DataLoader or `workers_per_gpu` setting. Changing a Python worker count therefore cannot alter those C++ end-to-end measurements; parallel image decoding would require a separate C++ pipeline change and a new timing protocol.

## Resume procedure

1. Trace the FP32 planning delta against PyTorch before attributing accuracy loss to reduced precision; prioritize export rewrites, plugin parity, temporal-state decoding, and result reconstruction.
2. Expand the ONNX output contract and add TensorRT result-bundle reconstruction if full detection/tracking/map/occupancy scores are required from the engine path; the NVIDIA tutorial graph cannot emit all of them.
3. Replace the in-memory monolithic calibration collector with a bounded-memory streaming or sharded protocol before expanding UniAD-base calibration; one 8-sample NPZ is already about 1.2 GiB.
4. Rebuild and rerun INT8 after the larger representative calibration is available; retain the current `calib8` lineage for comparison.
5. Install the complete official Bench2Drive/CARLA scenario assets before requesting a formal closed-loop driving score; retain the current Town03 result only as a custom diagnostic.
6. Update this document and push one commit after each completed major round.

---

## Iteration 008 - 2026-08-11T08:10:51-07:00

- Repaired initial temporal padding and CUDA resize parity, added strict scene resets, and rebuilt/reran the trained epoch-20 tiny graph with exact TensorRT 10.7.
- Completed 6018/6018 frames for FP32, FP16, and INT8(EQ)+FP16 with finite planning output and native mean/p50/p99 enqueue, synchronized forward, and end-to-end latency.
- Verified same-checkpoint FP32 planning parity (`0.780420 m` vs PyTorch `0.780738 m`) and the open-loop enqueue trend `17.371 > 13.043 > 12.889 ms`.
- Added persistent TensorRT and PyTorch CARLA services, then completed all four 1200-frame Town03 model-controlled diagnostics on identical sensors/controller inputs.
- Kept the CARLA result fail-closed as a custom diagnostic because complete Bench2Drive assets and the official scoring protocol are unavailable.
- Retained small source-only summaries under `UniAD/evidence/trained_tiny_exact107_full6018` and `UniAD/evidence/carla_closed_loop_town03_full1200`; checkpoints, ONNX, engines, raw frames, and logs remain excluded.

---

## Iteration 007 - 2026-08-10T20:19:06-07:00

- Verified the trained tiny pipeline against every training/export command in `documents/train_export.md`; export uses the real epoch-20 checkpoint and `tiny_imgx0.25_e2e_trt_p.py`.
- Unified `UniAD/ckpts` and `UniAD_train/ckpts` through one checkpoint registry while retaining the old `/data/ckpts` link as `ckpts.data-original`.
- Recorded the complete 6019-frame trained-tiny PyTorch baseline and the real-weight ONNX, 64-sample calibration, explicit-QDQ validation, and three engine builds.
- Measured all three engines with 100-iteration `trtexec`; FP16 is faster than INT8 on this RTX 4090, unlike the NVIDIA Orin-X ordering.
- Proved that changing the external profile optimum from 901 to 1150 does not remove the recurrent 10-second spikes; the remaining blocker is graph-internal active-track DDS/Myelin reconfiguration, not CPU fallback or an external profile miss.

---

## Iteration 006 - 2026-08-09T22:31:24-07:00

- Audited the working base deployment against every patch family required by NVIDIA's project-setup and explicit-quantization tutorials.
- Confirmed that base uses the official TensorRT-compatible module stack and a base-specific architecture/profile overlay; it was not exported through the tiny graph.
- Localized the acceptance failure to missing base export/runtime parity rather than missing patches or INT8 calibration alone, because FP32 already has a large trajectory delta.
- Confirmed that UniAD Python configurations already use eight DataLoader workers and documented that the C++ TensorRT evaluator has no Python DataLoader.

---

## Iteration 005 - 2026-08-09T20:58:49-07:00

- Completed FP16 and preliminary 8-sample-calibrated INT8 fixed-1150 evaluations over all 6018 frames.
- Verified exactly 6018 unique, strictly ordered frame records for each precision, finite planning/latency JSON, and no missing frames.
- Confirmed the full enqueue trend `FP32 185.945 ms > FP16 108.196 ms > INT8 97.302 ms`.
- Recorded the unresolved FP32 export/runtime accuracy gap and the additional INT8 collision degradation instead of treating latency success as accuracy reproduction.
- Retained source-only FP16 and INT8 evidence under `UniAD/evidence/uniad_base_e2e`; ONNX, engine, checkpoint, raw predictions, and calibration tensors remain excluded.

---

## Iteration 004 - 2026-08-09T19:45:35-07:00

- Completed all 6018 frames of the base FP32 fixed-1150 evaluation on GPU 6.
- Verified schema-v2 latency output, fixed input count `1150`, finite JSON values, and correct base `200x200` planning bounds.
- Confirmed exact planning-metric equivalence to the prior dynamic run while eliminating every second-scale Myelin shape-update spike.
- Started the INT8(EQ)+FP16 fixed-1150 full run on GPU 6 with an explicit `calib8` tag; this is a preliminary calibration result and will not be presented as a larger calibration protocol.

---

## Iteration 003 - 2026-08-09T19:22:25-07:00

- Added a strict runtime check that terminates with the exact frame index if any decoded planning coordinate is non-finite.
- Kept the fixed-1150 FP32 full validation running on GPU 6; its stable enqueue timing continues without the earlier dynamic-shape Myelin spikes.
- Started the fixed-1150 FP16 full 6018-frame validation on GPU 7 after the UniV2X INT8 and CARLA runs released that GPU.
- Continued to inherit the active PDT process timezone; no experiment script forces Shanghai time.

---

## Iteration 002 - 2026-08-09T18:56:27-07:00

- Added base-aware planning bounds and a segmentation-shape guard; the base evaluator now explicitly uses a `200x200` grid with `[-50, 50, 0.5]` bounds.
- Added native p99 output for model enqueue, synchronized inference call, and end-to-end latency.
- Added optional fixed temporal-track input padding using the ONNX graph's existing `-10000` invalid-row sentinel.
- Verified fixed 1150 input against dynamic input for 50 frames: every planning coordinate was exactly equal, with maximum absolute delta `0`.
- Fixed-input FP32 removed the Myelin shape-update spikes: enqueue mean/p50/p99 was `184.742/184.028/191.407 ms`.
- Started the fixed-1150 FP32 full 6018-frame rerun on GPU 6.
- Removed forced Shanghai timezone exports so experiment processes inherit the active PDT timezone.

---

## Iteration 001 - 2026-08-09T18:33:26-07:00

- Created the publication branch and source-only repository layout.
- Imported the UniAD export/quantization/evaluation scripts, deployment model changes, enqueueV3 runtime source, and small audit reports.
- Recorded the original-checkpoint PyTorch baseline, base engine smoke trend, active FP32 full evaluation, and remaining full-validation/CARLA work.
- Excluded weights, ONNX graphs, engines, datasets, calibration tensors, timing caches, compiled libraries, and large generated outputs.
