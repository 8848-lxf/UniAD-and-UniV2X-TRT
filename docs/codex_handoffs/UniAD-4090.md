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
- TensorRT is intentionally split by tutorial stage: x86 explicit-QDQ quantization/plugin work uses 10.9.0.34 from `/home/lixingfeng/uniad-trt/TensorRT-10.9_x86_cu118`; NVIDIA's published DRIVE Orin-X engine-build/runtime stage uses 10.7, so the already-serialized trained-tiny engines continue with their matching 10.7 runtime under `/data/lxf/uniad_deployment_outputs/toolchains`
- ModelOpt 0.29 source/editable target: `/home/lixingfeng/UniAD_examine/HEAL/prune_model/Model-Optimizer-0.29.0`, restored from the existing local `/home/lixingfeng/uniad-trt/uniad-2.0/TensorRT-Model-Optimizer-release-0.29.0` copy without reinstalling the conda environment
- GPU used for current full engine evaluation: RTX 4090, logical GPU 6

## Completion matrix

| Item | State | Evidence/notes |
| --- | --- | --- |
| NVIDIA tiny/random ONNX reference flow | Completed as a deployment smoke | All three precisions ran; random weights make its planning accuracy non-physical and unsuitable for accuracy reproduction. |
| Trained UniAD-tiny stage 1/2 | Completed | Stage 1 epoch 6 and stage 2 epoch 20 follow `documents/train_export.md`; both training and deployment `ckpts` entries resolve to one registry. |
| Trained UniAD-tiny PyTorch full validation | Completed | 6019 frames; detection, tracking, map, occupancy, planning, and latency are recorded below. |
| Trained UniAD-tiny ONNX/calibration/engines | Completed with corrected validation315 calibration | The old 64-sample graph is superseded. The replacement uses 315 independent recurrent validation feeds, bounded entropy histograms, and a fresh exact-TensorRT-10.7 static-1600 engine. |
| Trained UniAD-tiny temporal TensorRT evaluation | Completed for FP32/FP16/corrected INT8 | All three precisions completed 6018 frames with finite output, scene reset, fixed 1600 capacity, planning metrics, and mean/p50/p99 latency. |
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
| CARLA closed-loop evaluation | Route-conditioned INT8 full-route diagnostic completed; four-backend historical gate retained | The corrected INT8 UniAD-tiny route reaches the Town03 destination with zero collisions. The earlier four-backend model-controlled run remains a historical blocked diagnostic; neither protocol is an official Bench2Drive score. |

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

The original exported real-weight graph used a 64-sample training calibration package at `trained_tiny_epoch20/calibration/calib_data_shape0_901.npz`. It is superseded: ModelOpt 0.29.0 constructs its fixed-shape feed list with `[{}] * n_itr`, so all list entries alias one dictionary and the calibration loop repeatedly receives the final frame.

The corrected protocol sequentially processed all 6019 validation frames with the epoch-20 checkpoint and recursive track/BEV/timestamp/ego-pose state, resetting at 150 scene boundaries. It selected all 315 frames whose input track count is exactly 901 and wrote `calibration_validation_fixed/calibration/calib_data_shape0_901.npz`. Audits prove 315 unique NPZ frames and 315 independent provider feeds; the unmodified upstream provider exposes only one effective feed. The repository-local provider fixes the list allocation and fails closed unless every frame signature is unique, without modifying the installed ModelOpt environment.

The explicit-QDQ graph is mixed precision rather than literal every-operator INT8. The validation315 graph contains 377 `QuantizeLinear`, 526 `DequantizeLinear`, and 675 INT8 initializers. INT8 weights cover all 104 Conv nodes and 45 of 53 Gemm/linear nodes; all 504 MatMul nodes are intentionally excluded. Activation QDQ boundaries also reach Add, Resize, MaxPool, Mul, transpose/reshape-related paths, and some MatMul inputs, but QDQ adjacency alone does not prove that TensorRT selected an INT8 kernel for every such layer.

The first corrected entropy attempt exposed a second tool defect: 14 ORT histograms expanded above 2048 bins and the largest reached 507,826 bins. ORT's Python KL search is approximately quadratic in the expanded bin count and remained active after 3.6 hours. The accepted run symmetrically pads and exactly sums contiguous bins to at most 2048 while preserving every histogram count and full observed range, then calls the unchanged ORT entropy/KL threshold search. This local robustness patch is evidence-backed but is not byte-identical to unmodified upstream ModelOpt. The accepted quantization completed in 22.7 minutes, and TensorRT 10.7 parsed all 29,060 network layers and built the 157,675,380-byte engine in 459.3 seconds.

| `trtexec`, 100 iterations at track shape 901 | FP32 | FP16 | INT8(EQ)+FP16 |
| --- | ---: | ---: | ---: |
| GPU compute mean / p50 / p99 | 13.507 / 13.332 / 17.194 ms | 9.471 / 9.417 / 11.166 ms | 10.008 / 9.917 / 12.040 ms |
| overall mean / p50 / p99 | 14.675 / 14.502 / 18.362 ms | 10.764 / 10.709 / 12.521 ms | 11.185 / 11.098 / 13.250 ms |

The table above is retained as the original `trtexec` diagnostic. It used the old shape-901 standalone contract, where FP16 was faster than INT8 on this RTX 4090. The corrected temporal application supersedes it for end-to-end deployment evaluation.

### Corrected trained-tiny temporal deployment, full 6018 frames: raw TensorRT trajectory

The accepted rerun uses the TensorRT 10.7 runtime matching NVIDIA's published DRIVE Orin-X engine/runtime stage and the TensorRT version that serialized these local engines. The earlier x86 explicit-QDQ generation used ModelOpt 0.29 with TensorRT 10.9 as required by the tutorial. It also uses OpenCV-compatible half-pixel image resize, real `-10000` padding of the initial temporal tensors, fixed 1300/1600 capacities, and per-scene reset metadata. All 6018 frames completed with finite output.

| Metric | Deployment PyTorch raw | TensorRT FP32 raw | TensorRT FP16 raw | TensorRT INT8(EQ)+FP16 raw |
| --- | ---: | ---: | ---: | ---: |
| planning avg. L2 | 0.780738 m | 0.780420 m | 0.758038 m | 0.763937 m |
| planning point collision | 0.085854% | 0.085854% | 0.091392% | 0.088623% |
| planning box collision | 0.667442% | 0.667442% | 0.631439% | 0.800377% |
| NVIDIA `planning MSE` (mean point L2) vs PyTorch | 0 | 0.003772 m | 0.119177 m | 0.183680 m |
| literal coordinate MSE vs PyTorch | 0 | 2.2955e-05 m2 | 0.170937 m2 | 0.177275 m2 |
| enqueue mean / p50 / p99 | n/a | 17.371 / 17.258 / 19.669 ms | 13.043 / 12.907 / 16.896 ms | 12.041 / 12.110 / 13.879 ms |
| synchronized forward mean / p50 / p99 | n/a | 21.141 / 20.984 / 23.884 ms | 17.027 / 16.859 / 23.287 ms | 15.992 / 16.000 / 18.088 ms |
| end-to-end mean / p50 / p99 | n/a | 98.306 / 97.042 / 112.049 ms | 93.600 / 92.367 / 107.090 ms | 92.788 / 91.680 / 109.152 ms |

The table above is explicitly the raw TensorRT `outs_planning` output. Its latency JSON records `collision_optimization_enabled=false`; therefore its collision columns must not be compared directly with the original Python model's `use_col_optim=True` output. The enqueue/forward ordering is `FP32 > FP16 > INT8`, matching the NVIDIA example qualitatively. Relative to FP32 enqueue, FP16 is `1.33x` faster and INT8 is `1.44x` faster; INT8 is `1.08x` faster than FP16 on RTX 4090. Synchronized forward gives `1.32x` FP32-to-INT8 and `1.06x` FP16-to-INT8 speedups.

The local tutorial text defines its legacy `planning MSE` column as the average Euclidean L2 distance between corresponding TensorRT and PyTorch trajectory points, despite the MSE label. Our evaluator therefore computes `mean(norm(delta, axis=-1))` and reports coordinate-wise squared MSE and mean squared point-L2 separately. The public repository does not expose the script that generated the legacy table, so the numerical table cannot prove a hidden squaring step. For FP32, the four auditable aggregates are mean point L2 `0.003772 m`, coordinate MSE `2.2955e-05 m2`, mean squared point L2 `4.5909e-05 m2`, and square-of-mean L2 `1.4229e-05 m2`; they are not interchangeable.

The remaining planning gap has two distinct causes. FP32 is already `0.003772 m` rather than `9.2417e-07 m`, so it cannot be repaired by calibration; a frame-0 dynamic-901 probe (`0.006211 m`) matched the fixed-capacity frame-0 result, ruling out static padding as the primary cause. Reduced precision is dominated by scene-initial state sensitivity: across 150 scene starts, FP16 is `3.0777 m` on scene-first frames but `0.04355 m` after removing them (close to NVIDIA `0.0458 m`); INT8 is `3.0398 m` and `0.11067 m`, respectively. The raw full-sequence values remain `0.119177/0.183680 m`, so the accepted report does not claim full official parity. The audit is retained at `UniAD/evidence/trained_tiny_exact107_full6018/planning_metric_protocol_audit.json`.

### Corrected trained-tiny temporal deployment, full 6018 frames: `use_col_optim=True` protocol

The original UniAD evaluation config sets `use_col_optim=True`. The TensorRT graph emits raw `outs_planning`, so the C++ runner now optionally reconstructs occupied BEV points from `seg_out` and applies occupancy-aware collision trajectory optimization after inference. The following table was rerun with `UNIAD_COLLISION_OPTIMIZATION=1`; each latency JSON records `collision_optimization_enabled=true`.

| Metric | Full Python reference (`use_col_optim`) | TensorRT FP32 optimized | TensorRT FP16 optimized | TensorRT INT8(EQ)+FP16 optimized |
| --- | ---: | ---: | ---: | ---: |
| planning avg. L2 | 0.826565 m | 0.818865 m | 0.796663 m | 0.798525 m |
| planning point collision | 0.013847% | 0.036003% | 0.038773% | 0.044312% |
| planning box collision | 0.221558% | 0.340645% | 0.301872% | 0.324028% |
| NVIDIA `planning MSE` vs full Python optimized trajectory | 0 | 0.188433 m | 0.198145 m | 0.257322 m |
| synchronized forward mean/p50/p99 | n/a | 20.356/20.275/23.352 ms | 16.287/16.228/17.352 ms | 16.114/15.887/17.708 ms |
| end-to-end mean/p50/p99 | n/a | 96.504/95.228/111.030 ms | 92.497/91.109/108.303 ms | 91.482/90.876/103.718 ms |

The full Python `results.pkl` contains the collision-optimized output produced by the epoch-20 checkpoint. The first 6018 trajectories were extracted and checked against the evaluator ground truth with maximum alignment delta `0`, creating the missing same-protocol reference. This resolves the main collision inflation: raw FP32 box collision `0.667442%` falls to `0.340645%`; FP16 and INT8 are `0.301872%` and `0.324028%`, close to NVIDIA's `0.27%`. Average L2 is also in the official range, but planning MSE is not reproduced: local `0.188433/0.198145/0.257322 m` remains materially above NVIDIA's `9.2417e-7/0.0458/0.0502 m`. The remaining temporal/export/optimizer delta is therefore still an open accuracy-parity defect rather than a calibration-only issue.

### CARLA Town03 model-controlled closed-loop diagnostic (historical, superseded)

The earlier rerun uses one dense route and model-controlled steering. Its blocked results are retained for root-cause history only. The route-conditioned controller below supersedes it: the global route target controls steering, the model trajectory controls speed, and explicit forward gear is set for the reused CARLA 0.9.10.1 binary.

| Backend | Frames / result | Route progress | Unique collisions / lane invasions | Forward mean / p50 / p99 | Service E2E mean / p50 / p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP32 | 587 / blocked | 1.5014% | 22 / 0 | 98.844 / 94.146 / 146.646 ms | 270.997 / 267.954 / 345.994 ms |
| TensorRT FP32 | 582 / blocked | 1.5014% | 22 / 0 | 22.958 / 22.747 / 30.210 ms | 205.325 / 203.502 / 261.155 ms |
| TensorRT FP16 | 585 / blocked | 1.5861% | 22 / 0 | 18.419 / 16.165 / 36.109 ms | 193.770 / 192.949 / 223.861 ms |
| TensorRT INT8(EQ)+FP16 | 904 / blocked | 1.2936% | 38 / 0 | 19.185 / 18.379 / 29.780 ms | 192.501 / 197.121 / 241.740 ms |

These rows document the previous single-target controller failure and are not used as the final INT8 CARLA result.

The reused V2Xverse asset tree contains a 42-route catalogue with 4829 scenario trigger-event configurations, plus 233 training-split, 105 evaluation, and 66 additional route definitions. This evidence executed only `routes_town03_1.xml` and spawned zero scenario events because the custom runner bypasses Leaderboard/ScenarioRunner. It can report progress, collision groups, lane invasions, blocked reason, speed, and latency, but not a valid official route completion or Driving Score until the model is wrapped as a leaderboard agent and run through that evaluator.

### CARLA Town03 route-conditioned INT8 full-route rerun

The corrected route protocol uses the same dense 1171-waypoint, 1181.667 m Town03 route, six synchronized RGB cameras, no traffic, 0.05 s synchronous ticks, inference every 10 frames, global-route steering, model-trajectory speed control with EMA `0.35`, and a 500-frame blocked threshold. It reaches the destination without collision; lane invasion is reported separately. The service uses one TensorRT execution context and fixed track capacity, so no per-frame shape profile rebuild is involved.

| Backend | Frames / result | Progress / final distance | Collision / lane invasion | Forward mean / p50 / p99 | Service E2E mean / p50 / p99 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TensorRT INT8(EQ)+FP16 | 6134 / destination reached | 99.5845% / 4.944 m | 0 / 29 | 15.510 / 14.414 / 24.902 ms | 190.216 / 189.562 / 212.623 ms |

The first long-route run stopped at frame 1850 because `prev_track_intances1_out` became non-finite. This was not OOM: the CARLA server and CUDA context were alive, and the service returned a finite planning tensor while a recurrent tracking tensor contained two invalid values. The service now discards only invalid recurrent outputs, resets the temporal state, records the recovery, and continues; the accepted rerun recovered once at frame 1860 and completed the route with zero collisions. The raw result is in `/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/carla_route_conditioned_v4_temporal_recovery_fullroute6500_20260812/tensorrt_int8/closed_loop_metrics.json`, with a source-only summary at `UniAD/evidence/carla_route_conditioned_int8_fullroute/summary.json`. This custom route protocol does not provide a Bench2Drive Driving Score.

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
3. If exact collision parity is required, align the C++ optimizer numerics and occupancy-boundary convention with the Python CasADi implementation; the missing raw-trajectory post-process has now been restored and separately measured.
4. Replace the in-memory monolithic calibration collector with a bounded-memory streaming or sharded protocol before expanding UniAD-base calibration; one 8-sample NPZ is already about 1.2 GiB.
5. Wrap the post-processed backend as a Leaderboard/ScenarioRunner agent and validate the existing route/scenario catalogues before requesting a formal driving score; retain the current Town03 result only as a custom diagnostic.
6. Update this document and push one commit after each completed major round.

---

## Iteration 013 - 2026-08-12T21:08:00-07:00

- Audited the planning metric contract. The local official tutorial says `planning MSE` is mean trajectory-point Euclidean L2; coordinate MSE, mean squared point-L2, and square-of-mean L2 are now emitted separately. A 150-scene audit found FP16 `0.04355 m` after excluding scene-first frames versus full-sequence `0.119177 m`; INT8 was `0.11067 m` versus `0.183680 m`. The public source does not expose the legacy table generator, so a hidden squared interpretation is not claimed.
- Added route-conditioned steering and model-speed control to the CARLA runner. The first full INT8 run exposed long-route recurrent-state non-finites at frame 1850; this was not OOM. The service now resets only invalid recurrent state, records the event, and continues with finite planning output.
- Completed UniAD-tiny INT8 Town03 route: `6134` frames, `99.5845%` progress, destination reached at `4.944 m`, zero collisions, 29 lane invasions, one temporal-state recovery, forward `15.510/14.414/24.902 ms`, service E2E `190.216/189.562/212.623 ms` (mean/p50/p99).
- Kept the earlier four-backend blocked CARLA table as historical; it bypassed dense-route conditioning and is not a final route-completion result.

## Iteration 012 - 2026-08-12T19:17:28-07:00

- Clarified the two-stage NVIDIA version contract from `documents/explicit_quantization.md`: x86 explicit-QDQ quantization and plugin compilation use ModelOpt 0.29 with TensorRT >=10.9, while the published DRIVE Orin-X engine build/runtime and result table use TensorRT 10.7.
- Stopped treating the trained-tiny 10.7 runtime as the x86 quantization toolchain. Existing 10.7 engines remain on their serialization-compatible runtime; `/home/lixingfeng/uniad-trt/TensorRT-10.9_x86_cu118` remains the active x86 quantization/build package.
- Recovered the deleted ModelOpt editable source target by copying the existing local 0.29 source, including `nvidia_modelopt.egg-info`, to `/home/lixingfeng/UniAD_examine/HEAL/prune_model/Model-Optimizer-0.29.0`. No conda package was installed or modified; direct import reports 0.29.0 and the quantization CLI help probe passes.

---

## Iteration 011 - 2026-08-12T19:05:13-07:00

- Extracted 6018 full-Python `use_col_optim=True` trajectories from the epoch-20 `results.pkl` and verified ground-truth row alignment with maximum absolute delta `0`.
- Recomputed same-protocol planning metrics. FP32/FP16/INT8 box collision is `0.340645/0.301872/0.324028%`, while planning MSE against the optimized Python reference is `0.188433/0.198145/0.257322 m`; collision is close to NVIDIA, but planning MSE is not reproduced.
- Added occupancy-aware collision post-processing to both TensorRT and deployment-PyTorch CARLA services, including separate post-process latency and activation statistics.
- Isolated the initial CARLA no-motion defect with a physics probe: automatic direct control remained in neutral (`gear=0`), while explicit forward gear moved the vehicle. Added `manual_gear_shift=true, gear=1` to both model controllers.
- Completed the corrected PyTorch/FP32/FP16/validation315-INT8 Town03 diagnostic. All backends start and move, then still terminate as blocked after early contact; the common failure remains a custom sensor/domain/controller integration issue rather than an INT8-only failure.

---

## Iteration 010 - 2026-08-12T17:32:08-07:00

- Verified the trained tiny deployment checkpoint by SHA-256: epoch-20 tiny is `88289ecdf2a7576a4c6bd1c0879d15ffba5b93fc22221872fef6196638d91484`; the separate base checkpoint is `0ad0c2f5...` and was not used by the tiny engine.
- Audited the C++ runner's latency JSON and found all earlier tiny full-validation rows had `collision_optimization_enabled=false`; they were raw trajectory statistics, unlike the Python config's `use_col_optim=True` path.
- Rebuilt no engine and changed no weights. Reran FP32, FP16, and validation315 INT8 on all 6018 frames with `UNIAD_COLLISION_OPTIMIZATION=1`, restoring occupancy-aware collision post-processing after raw TensorRT output.
- Corrected optimized box collision is `0.340645% / 0.301872% / 0.324028%` for FP32/FP16/INT8, versus raw `0.667442% / 0.631439% / 0.800377%`; optimized values are close to NVIDIA's `0.27%` and expose the remaining optimizer/occupancy-boundary/statistical differences.
- Added `collision_optimized_summary.json` and split the report into raw and optimized protocol tables. All three optimized runs completed 6018/6018 finite frames.
- Made the exact-TensorRT-10.7 evaluation entry point default to the optimized protocol, while retaining `UNIAD_COLLISION_OPTIMIZATION=0` as an explicit raw diagnostic override.
- Audited CARLA assets and corrected the earlier overstatement: route/scenario files exist, but the retained custom run uses one route, zero scenario events, raw planning, and no official scorer.

---

## Iteration 009 - 2026-08-12T15:42:57-07:00

- Proved and fixed ModelOpt 0.29's fixed-shape feed aliasing: the 4.99 GiB validation package contains 315 unique recurrent shape-901 frames, while upstream exposed one repeated final-frame feed.
- Added bounded, count-preserving entropy histograms after a probe found 507,826-bin ORT expansion; completed the corrected quantization in 22.7 minutes without changing the installed ModelOpt environment.
- Built a fresh exact-TensorRT-10.7 static-1600 engine and completed 6018/6018 finite frames with INT8 planning L2 `0.763937 m`, box collision `0.800377%`, planning-to-PyTorch L2 `0.183680 m`, and forward `15.992/16.000/18.088 ms`.
- Completed the corrected INT8 Town03 diagnostic: 571 frames, blocked at `2.3791%`, 19 unique collisions, 6 lane invasions, and steady-state engine forward `19.504/16.299/48.116 ms`.
- Recorded that calibration improved but did not close NVIDIA accuracy parity; the deployment collision baseline and full Python baseline remain inconsistent and require separate GT/evaluator tracing.

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
