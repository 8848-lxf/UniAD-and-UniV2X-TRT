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
- Local x86 primary chain: TensorRT 10.9.0.34 from `/home/lixingfeng/uniad-trt/TensorRT-10.9_x86_cu118` for explicit-QDQ quantization, plugin compilation, engine build, runtime, and evaluation. NVIDIA's published DRIVE Orin-X engine-build/runtime table uses TensorRT 10.7; local 10.7 artifacts under `/data/lxf/uniad_deployment_outputs/toolchains` are historical controls only.
- ModelOpt 0.29 source/editable target: `/home/lixingfeng/UniAD_examine/HEAL/prune_model/Model-Optimizer-0.29.0`, restored from the existing local `/home/lixingfeng/uniad-trt/uniad-2.0/TensorRT-Model-Optimizer-release-0.29.0` copy without reinstalling the conda environment
- GPUs used for the current parallel full engine evaluation: RTX 4090, logical GPUs 4-7

## Completion matrix

| Item | State | Evidence/notes |
| --- | --- | --- |
| NVIDIA tiny/random ONNX reference flow | Completed as a deployment smoke | All three precisions ran; random weights make its planning accuracy non-physical and unsuitable for accuracy reproduction. |
| Trained UniAD-tiny stage 1/2 | Completed | Stage 1 epoch 6 and stage 2 epoch 20 follow `documents/train_export.md`; both training and deployment `ckpts` entries resolve to one registry. |
| Trained UniAD-tiny PyTorch full validation | Completed | 6019 frames; detection, tracking, map, occupancy, planning, and latency are recorded below. |
| Trained UniAD-tiny ONNX/calibration/engines | Completed on TensorRT 10.9 | The accepted INT8 graph uses 168 independent official-literal recurrent feeds and protects only two occupancy-terminal activation Q/DQ paths with FP16. The 315-feed graph is retained as a calibration-semantics control. |
| Trained UniAD-tiny temporal TensorRT evaluation | Completed for TensorRT 10.9 FP32/FP16/accepted INT8 | All three precisions completed 6018 frames with official external-state carry semantics, historical fixed-1600 capacity, raw/optimized planning metrics, collision-trigger audits, and mean/p50/p99 latency. A later profile audit found max track count 1138 and mandates official fixed-1150 capacity for future formal rebuilds. |
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

## 2026-08-13T21:47:23-07:00 (PDT) - CV-JPEG and full official-literal parity round

- Replaced the runtime STB JPEG decode path with Conda libjpeg-turbo and verified the first six camera tensors against PyTorch: max normalized delta `2.3841858e-7`.
- Added explicit `official_literal` / `scene_reset` protocol manifests, recurrent PyTorch audit output, occupancy packbits comparison, and three-way planning statistics.
- Re-ran 6018 frames with TensorRT 10.9 FP32/FP16/accepted INT8 and a matching 6018-frame PyTorch reference. FP32/FP16/INT8 optimized box Col is `0.252022% / 0.459732% / 0.263100%`; raw-vs-PyTorch squared-point planning MSE is `6.432e-7 / 0.061875 / 0.080659`.
- Confirmed the previous all-zero INT8 occupancy branch was caused by two terminal activation Q/DQ scale paths feeding a threshold, not by two Mul operators. The accepted graph protects those activation edges with FP16; all three engines now modify trajectories on nonzero occupancy frames.
- Changed `evaluate_planning_outputs.py` schema 2 so the primary `planning_mse` is explicitly `mean(dx^2+dy^2)` while mean point L2 and coordinate MSE remain separately named.
- Fixed runtime/build scripts that referenced a missing `repro/package/uniad-trt` tree. CMake now builds the checked-in runtime with explicit Conda dependency roots and the runner starts from `UniAD/repro/UniAD_deploy` so relative `data/...` image paths resolve.
- Compact result evidence is stored in `UniAD/evidence/trained_tiny_cvjpeg_official_literal_full6018/summary.json`; large engines, ONNX files, checkpoints, and datasets remain outside Git.

---

## 2026-08-14T08:43:36-07:00 (PDT) - FP16 occupancy mixed-precision closure

- Resumed the interrupted `fp16_all_state_fp32` run and evaluated all 6018 frames. Protecting recurrent BEV and float track-state lineages improved occupancy IoU only from `71.278528%` to `71.686034%`; optimized avg. L2 regressed from `0.837544` to `0.842598 m`, and optimized box Col changed from `0.459732%` to `0.462501%`. This candidate is rejected.
- Scene-position aggregation localized the effect. Recurrent-state protection raises scene-start IoU from `79.749421%` to `92.783810%`, but offsets `1-4`, `5-9`, `10-19`, and `20+` remain `75.295846/69.902859/69.975025/71.572604%`. The dominant error is recomputed every frame in the shared feature and occupancy consumer path; it is not only long-horizon state accumulation.
- Fixed the mixed-precision builder audit boundary. `--fp32-lineage-convolutions` now distinguishes FP32 output dtype from FP32 convolution compute and records convolution depth. `--force-fp32-exclusive-output-lineage` records and constrains only a requested output group's non-shared ancestry. Both parameters default off.
- A controlled 200-frame sweep compared one Python-builder FP16 control with final-Conv, decoder-Conv-depth `2/4/8`, recurrent-state, occupancy/planning-exclusive, and combined state+consumer candidates. The best IoU was `92.017620%` for the combined candidate, only `0.009589` point above state-only, while enqueue p50 increased from `11.984384` to `14.986192 ms`. No candidate passed a full-run promotion gate.
- Added optional `seg_score_out` binding support and `UNIAD_DUMP_OCCUPANCY_SCORES=1`. A debug engine exposed the exact `ReduceMax(pred_ins_sigmoid)` values before `Greater(..., 0.1)`; `score > 0.1` reproduced `seg_out` bit-for-bit on 200/200 frames. The globally optimal threshold was about `0.0990` and improved IoU by only `0.013249` point. Threshold bias is not the root cause.
- 3x3 opening, closing, and median filters reduced IoU to `88.936263/84.492192/89.174446%`; FP16 errors are not removable isolated speckles.
- Final acceptance: plain FP16 remains a raw-planning diagnostic and is rejected for the collision-optimized deployment path. Use TensorRT FP32 for the accuracy reference or the already accepted explicit-INT8 graph with terminal occupancy FP16 protection for reduced precision, retaining its documented occupancy-parity limitation. Do not report ordinary FP16 as full-task accuracy-equivalent.
- Compact evidence: `UniAD/evidence/trained_tiny_fp16_occupancy_audit_20260814/summary.json`. Engines, timing caches, score dumps, packed masks, and reports remain under `/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/trt109_20260814` and are not committed.

---

## 2026-08-14T11:22:12-07:00 (PDT) - FP16 full-run promotion and threshold audit

- Promoted the best 200-frame mixed candidate to a 6018-frame fail-closed evaluation. The engine keeps LayerNorm plus recurrent BEV/float-track output lineages in FP32 and uses FP16 elsewhere. It completed all frames with finite output, but occupancy IoU was only `71.697673%`; optimized avg. L2/box Col was `0.842812 m / 0.462501%`. Model enqueue mean/p50/p99 was `15.791/15.789/17.703 ms`, slower than plain FP16. It is rejected.
- The same candidate reduced raw trajectory squared-point planning MSE against matching PyTorch from plain FP16 `0.061875` to `0.006976`, and mean point L2 from `0.032670` to `0.015541 m`. This verifies that recurrent/LayerNorm protection improves trajectory numerics, but it does not repair the separate occupancy consumer error.
- Added `UniAD/repro/scripts/sweep_fp16_occupancy_thresholds.py`. It validates score/raw temporal manifests, memory-maps the continuous score output, reuses the deployment-matching independent-point BFGS optimizer, and reports full GT L2/point/box-collision results for each threshold. A 200-frame `0.1` regression reproduced the C++ optimized metrics exactly.
- Repeated the 6018-frame score dump with the unmodified standard plugin (`SHA256 14ea7801...`) before the final threshold sweep. Default `0.1` gives `0.837544 m / 0.459732%`; validation-GT-selected `0.55` gives `0.745060 m / 0.354492%`. It improves this validation result but still misses FP32 `0.252022%` collision accuracy.
- Threshold `0.55` is diagnostic only. It was selected on validation GT, changes the trained/exported `score > 0.1` contract, and conflicts with the PyTorch occupancy-parity optimum near `0.099`; it is therefore not promoted into runtime or reported as a repaired model.
- Final acceptance is unchanged: plain FP16 and the tested mixed candidates are not accepted for collision-optimized deployment. FP32 remains the accuracy reference; the explicit-QDQ INT8 graph with terminal occupancy protection remains the reduced-precision result, with its residual occupancy-parity limitation stated.

---

## Iteration 014 - 2026-08-13T08:19:03-07:00

- Added an auditable `official_literal` calibration mode matching NVIDIA's script semantics: only global `sample_id == 0` initializes external recurrent state; scene changes carry the previous external track/BEV/timestamp/pose outputs and signal the model-internal reset with `use_prev_bev=0`. The existing per-scene reset remains the default for compatibility.
- Fixed only the two known saving defects: calibration batches are accumulated instead of overwriting `npz_data`, and the sample counter is not reset inside the loop. The 24-input exported-graph schema is unchanged.
- Sequentially processed all 6019 validation frames with the trained tiny epoch-20 checkpoint. Official-literal shape 901 selected 168 frames: 5 with `use_prev_bev=0` and 163 recursive frames. All 168 NPZ signatures and all 168 provider dictionaries are independent and unique.
- Compared calibration sets: official-literal 168 and scene-reset 315 intersect on 160 frames; 8 are literal-only and 155 are reset-only. This proves that the earlier 315 count is primarily a temporal-protocol difference, not an NPZ-saving bug.
- Completed bounded entropy quantization with ModelOpt 0.29, TensorRT 10.9 calibration EP, `dq_only`, global MatMul weight exclusion, and `max_bins=2048`. Twelve expanded histograms were symmetrically rebinned with exact count preservation. The graph has 460 reported quantized nodes: all 104 Conv, 45 Gemm, no MatMul INT8 weights, and 71 MatMul activation-QDQ adjacencies.
- Built a fresh static-1600 engine with isolated TensorRT 10.7/CUDA 11.8 and completed 6018/6018 finite frames. Artifact SHA256 values are recorded in `official_literal_calibration_full168_summary.json`.
- Added optional runtime `UNIAD_CARRY_STATE_ACROSS_SCENES=1`. It carries external outputs across scene boundaries while keeping `use_prev_bev=0`; the latency JSON records the protocol. Carry and scene-reset application outputs were bit-identical over all 6018 frames, confirming that the model-internal reset masks the carried external state at scene changes.
- Official-literal INT8 raw result: avg. L2 `0.764093 m`, box collision `0.695137%`, NVIDIA-defined planning MSE `0.181729 m`, coordinate MSE `0.177157 m2`. Enqueue/inference/E2E mean-p50-p99 are `11.640/11.749/13.391`, `15.567/15.622/17.587`, and `90.747/89.656/105.139 ms`.
- Accuracy parity is not accepted: NVIDIA INT8 reports avg. L2 `1.0029 m`, collision `0.27%`, and planning MSE `0.0502 m`. The local avg. L2 is better because the checkpoint differs, but collision and trajectory-to-PyTorch parity are materially worse. The official-literal calibration slightly improves planning MSE over scene-reset-315 (`0.181729` versus `0.183680 m`) while worsening optimized collision (`0.695137%` versus `0.324028%`).
- The H800-reported 196-feed natural/scene-reset package is not aligned with the literal NVIDIA temporal semantics. Its Conv `104/104`, Gemm `45/53`, and MatMul weight `0/504` coverage can be layer-aligned with this run, but its feed list must be regenerated and frame/feed hashes compared before cross-machine calibration parity can be claimed.

---

## Iteration 015 - 2026-08-13T11:55:00-07:00

This iteration supersedes the local TensorRT 10.7 trained-tiny engine/runtime rows as the RTX 4090 primary result. TensorRT 10.7 artifacts remain only as a historical, serialization-compatible control for NVIDIA's DRIVE Orin-X table. The active local chain is TensorRT `10.9.0.34` from `/home/lixingfeng/uniad-trt/TensorRT-10.9_x86_cu118`, conda CUDA `11.8.89`, and conda GCC `11.2.0`; `ldd` and CMake compiler detection prove that no system CUDA or system compiler was used.

NVIDIA's public table remains a TensorRT 10.7 DRIVE Orin-X reference:

| Framework / precision | Official latency | FPS | avg. L2 | avg. Col | planning MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| PyTorch 1.12 FP32 | 843.5172 ms | 1.18 | 0.9986 | 0.27% | 0 |
| TensorRT 10.7 FP32 | 64.0469 ms | 15.61 | 0.9986 | 0.27% | 9.2417e-7 |
| TensorRT 10.7 FP16 | 49.7559 ms | 20.10 | 1.0021 | 0.26% | 0.0458 |
| TensorRT 10.7 INT8(EQ)+FP16 | 39.3125 ms | 25.44 | 1.0029 | 0.27% | 0.0502 |

### Planning-difference metric audit

The public prose calls `planning MSE` the average pointwise Euclidean L2 distance between TensorRT and PyTorch trajectories. The name and the FP32 value `9.2417e-7`, however, resemble a squared-error statistic, and NVIDIA does not publish the table generator. The evaluator therefore no longer asserts that the metric is definitely not squared. It reports mean point L2, coordinate MSE, and mean squared point distance separately.

On byte-identical saved frame-5 inputs, TensorRT 10.9 FP32 versus PyTorch is:

| Statistic | Value |
| --- | ---: |
| mean point L2 | 2.763435e-4 m |
| coordinate MSE | 6.753211e-8 m2 |
| mean squared point distance | 1.350642e-7 m2 |

This proves that the earlier full-application FP32 point-L2 gap was dominated by preprocessing/recursive-runner protocol rather than the engine's same-input numerical error. It also confirms that a squared statistic is numerically much closer to NVIDIA's FP32 table value, but does not prove which squared reduction NVIDIA used.

### Calibration and occupancy root cause

The official-literal calibration scan processed 6019 frames, initialized external temporal state only at global sample zero, carried external state across scene changes, set `use_prev_bev=0` at scene changes, and selected 168 independent shape-901 feeds. The runtime audit confirms that `use_prev_bev=0` occurs only on 150 scene starts; 5868 later frames retain temporal propagation.

The unprotected 168-feed INT8 graph failed the occupancy branch: all 6018 `seg_out` tensors were zero, so collision optimization had zero candidate points and changed zero trajectories. This is why its raw and supposedly optimized outputs were byte-identical and its box collision stayed at `0.695137%`.

Backward scale tracing found the largest relevant discrepancy at the final occupancy product: `onnx::Mul_26440_scale` was `0.0004998153` in the official-literal graph versus `0.007513786` in the occupancy-valid 315-feed control, a factor of about 15.04. The accepted mixed-precision graph bypasses only the activation Q/DQ pairs on `onnx::Mul_26440` and `onnx::Mul_26447`; it removes four Q/DQ nodes while retaining 375 QuantizeLinear and 524 DequantizeLinear nodes. Conv/Gemm/other accepted explicit-QDQ coverage is unchanged.

A 200-frame gate recovered positive occupancy on 164 frames and modified 68 trajectories, versus FP32 165/70. The full 6018-frame accepted engine has positive occupancy on 4808 frames and modifies 2736 trajectories. The failed unprotected engine is retained only as negative evidence.

### TensorRT 10.9 full-validation result

All rows use one execution context, fixed capacity 1600, official external-state carry semantics, 150 scene-start `use_prev_bev=0` resets, and 6018 finite frames. The four full runs were executed concurrently on logical GPUs 4-7; their latency columns are complete per-run measurements but include shared CPU/I/O contention. `Raw` is direct `outs_planning`; `optimized` applies the audited occupancy-aware collision post-process from the same forward. Planning difference for raw output uses raw deployment PyTorch; optimized planning difference uses full Python `use_col_optim=True` output.

| Metric | PyTorch reference | TRT 10.9 FP32 | TRT 10.9 FP16 | TRT 10.9 INT8(EQ)+FP16 accepted |
| --- | ---: | ---: | ---: | ---: |
| raw avg. L2 | 0.780738 m | 0.780422 m | 0.758029 m | 0.764122 m |
| raw box Col | 0.667442% | 0.667442% | 0.628670% | 0.695137% |
| raw mean point L2 vs PyTorch | 0 | 0.003771 m | 0.119161 m | 0.179641 m |
| raw coordinate MSE vs PyTorch | 0 | 2.28579e-5 m2 | 0.170862 m2 | 0.177324 m2 |
| optimized avg. L2 | 0.826565 m | 0.819064 m | 0.796739 m | 0.801195 m |
| optimized box Col | 0.221558% | 0.343414% | 0.307411% | 0.332336% |
| optimized mean point L2 vs optimized Python | 0 | 0.188819 m | 0.198107 m | 0.253696 m |
| positive occupancy frames | n/a | 4823 | 4802 | 4808 |
| collision-optimizer modified frames | n/a | 2766 | 2744 | 2736 |
| model enqueue mean / p50 / p99 | n/a | 17.635 / 17.517 / 20.092 ms | 12.318 / 12.357 / 13.154 ms | 11.236 / 10.873 / 13.202 ms |
| synchronized forward mean / p50 / p99 | n/a | 21.616 / 21.430 / 25.307 ms | 16.315 / 16.301 / 17.398 ms | 15.095 / 14.766 / 17.260 ms |
| end-to-end mean / p50 / p99 | n/a | 104.780 / 104.185 / 113.197 ms | 98.507 / 97.822 / 108.146 ms | 96.045 / 95.329 / 106.012 ms |

For the official `trtexec --iterations=100` timing boundary at track shape 901, the three engines were rerun serially on the same logical GPU after 1000 ms warmup. GPU Compute Time mean/p50/p99 is FP32 `16.3179/16.2760/20.5210 ms`, FP16 `9.8777/9.8729/9.9471 ms`, and accepted INT8 `9.9588/9.9512/10.2083 ms`. FP32 to reduced precision speedup is reproduced. FP16 and INT8 are statistically tied on RTX 4090, so the Orin-X FP16-to-INT8 speedup is not reproduced on this hardware.

Acceptance is partial. Runtime, finite outputs, TensorRT 10.9 isolation, official calibration temporal semantics, occupancy validity, and collision post-processing pass. Optimized avg. L2 and collision are close in range to NVIDIA's example. Exact official planning-difference parity remains unresolved because the public metric reduction is ambiguous and the full recursive application still differs materially from the optimized Python trajectory.

Evidence: `UniAD/evidence/trained_tiny_trt109_full6018/summary.json`. Large ONNX, engine, timing-cache, prediction, and log artifacts remain excluded under `/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/trt109_20260813`.

---

## Iteration 016 - 2026-08-14T12:01:23-07:00

- Corrected the profile description. NVIDIA uses `MIN/OPT/MAX=901/901/1150`; the later 1600 setting was a conservative fixed-shape diagnostic bound, not an official requirement. Across the 6018 official-literal PyTorch audit frames, track count is `min=901`, `max=1138`, and `mean=926.580924`; no frame exceeds 1150. The controlled 1150/1600 200-frame run was byte-exact, so padding is not the FP16 occupancy defect. Historical 1600 results retain their label, while version-controlled build/evaluation defaults and future formal runs use 1150.
- Audited all `75,225,000` standard-plugin FP16 continuous occupancy scores before `Greater(..., 0.1)`. FP16 has 706,802 false-positive and 841,915 false-negative cells relative to matching PyTorch masks. False-positive score median is `0.229126`; finite false-negative median is `0.004032`. Only `0.2243%` of all scores lie within `0.1 +/- 0.01`, confirming that the defect is not threshold-near FP16 rounding.
- Found 31,700 NaN scores in only frames 1948, 1953, and 4825. They cover 2,854 PyTorch-positive cells and account for `0.338989%` of all false negatives. This is retained as a real FP16 numerical defect, but it cannot explain the aggregate occupancy/Col regression.
- Completed the mixed-precision path bisection. On the same 200-frame gate, FP16 control IoU is `91.598030%`; plugin FP32 accumulation `91.606901%`; MSDA FP32 `91.458717%`; final Conv FP32 `90.484706%`; occupancy decoder Conv depth 2/4/8 `90.538580/90.445100/91.345556%`; recurrent-state FP32 `92.008031%`; and recurrent state plus occupancy/planning-exclusive FP32 `92.017620%`. No terminal layer, plugin, threshold, or recurrent-state boundary restores FP32 occupancy parity.
- The remaining error boundary is the combined per-frame shared image/BEV feature producer and occupancy consumer propagation. Expanding a single-engine FP32 lineage further approaches FP32 compute scope without restoring the mask, so plain FP16 remains rejected for collision-optimized deployment. TensorRT FP32 is the accuracy reference; accepted explicit INT8 QDQ with terminal occupancy FP16 protection remains the reduced-precision path, with its residual occupancy-parity limitation documented.

Detailed PDT timeline: `/home/lixingfeng/UniAD_examine/DL4AGX/AV-Solutions/docs/4090-UniAD-tiny.md`. Large score dumps, engines, reports, and timing caches remain excluded under `/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/trt109_20260814`.

---

## Iteration 021 - 2026-08-14T23:11:14-07:00 (PDT)

### Collision optimizer parity and 3x3 cross-ablation

- Clarified why the deployment BFGS exists. NVIDIA's public Python tiny config sets `use_col_optim=True` and uses `CollisionNonlinearOptimizer` with CasADi/IPOPT, but the exported `forward_trt` returns raw planning and the public C++ README/sample explicitly leaves collision correction unimplemented. The local BFGS was added to fill that deployment gap; it is not an NVIDIA sample component.
- Added `UniAD/repro/scripts/audit_collision_optimizer_parity.py`. It read the existing 6018-frame FP32/FP16/INT8 raw trajectories and packed occupancy outputs without modifying engines, checkpoints, datasets, or configured environments. It completed `54,162` raw/occupancy frame tasks and `162,486` method/pair/frame results with 8 CPU workers in the existing `torch112` environment; all CasADi/IPOPT solves succeeded.
- Audited three distinct semantics: current floating-grid BFGS, public IPOPT with the same floating grid (solver-only control), and the literal public `planning_head.py` path. The literal path matters because `torch.nonzero()` returns `int64`, and assigning `(pixel-center)*0.5+0.25` back into that tensor truncates coordinates toward zero.

| Matched post-process | FP32 avg. L2 / box Col | FP16 avg. L2 / box Col | INT8 avg. L2 / box Col |
| --- | ---: | ---: | ---: |
| deployment BFGS, float grid | `0.776288 / 0.252022%` | `0.837544 / 0.459732%` | `0.784751 / 0.263100%` |
| IPOPT, same float grid | `0.776184 / 0.254791%` | `0.837380 / 0.459732%` | `0.784621 / 0.265869%` |
| public Python literal, int grid + IPOPT | `0.827332 / 0.229866%` | `0.913302 / 0.437576%` | `0.838548 / 0.218788%` |

- With identical floating coordinates, replacing BFGS by IPOPT changes box Col by only `+0.002769 / 0 / +0.002769` percentage points for FP32/FP16/INT8. The custom solver is therefore not the FP16 regression source. Rare non-convex local-solution outliers exist, but they do not explain aggregate Col.
- The public-Python integer coordinate behavior changes absolute L2/Col and must be retained when claiming parity with `use_col_optim=True`; it still leaves FP16 far above FP32 and INT8.

Public-Python IPOPT box Col cross-ablation, rows = raw trajectory and columns = occupancy:

| Raw \ occupancy | FP32 occupancy | FP16 occupancy | INT8 occupancy |
| --- | ---: | ---: | ---: |
| FP32 raw | `0.229866%` | `0.434807%` | `0.216019%` |
| FP16 raw | `0.227096%` | `0.437576%` | `0.213249%` |
| INT8 raw | `0.238174%` | `0.432037%` | `0.218788%` |

- Holding occupancy fixed makes raw precision almost irrelevant to collision. Holding raw fixed and selecting FP16 occupancy consistently raises box Col to `0.432-0.438%`; FP32/INT8 occupancy remain `0.213-0.238%`. This proves that global occupancy IoU is the wrong predictor: FP16 has `636,074` floating-grid candidates within 5 m of the raw path versus INT8 `513,075`, so its errors are more concentrated in the collision-sensitive corridor.
- The accepted conclusion is narrower and stronger: the FP16 Col anomaly is caused by the spatial distribution of FP16 occupancy near the planned path. It is not caused by BFGS versus IPOPT, raw planning precision, global occupancy IoU, profile 1600, threshold, or the old temporal protocol. NVIDIA's exact TRT table post-processing remains unpublished, so the local result must state which of the three post-process semantics it uses.

Compact evidence: `UniAD/evidence/trained_tiny_collision_optimizer_parity_full6018/summary.json`. Full resumable trajectories, statuses, and cross-ablation CSV are under `/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/collision_optimizer_parity_20260814/full6018`.

---

## Iteration 011 - 2026-08-12T22:24:00-07:00

- Re-checked `tools/prepare_calib_data.py` against the NVIDIA implementation. The retained semantics are unchanged: sequential validation traversal, per-scene temporal reset, recursive track/BEV/timestamp/ego-pose state, and cherry-pick only frames whose current `prev_track_intances0` count is exactly 901. The original implementation's per-loop `npz_data` overwrite/alias behavior is fixed; 6019 validation frames yielded 315 selected frames and 315 unique feed signatures.
- Audited the ModelOpt 0.29 calibration provider. The upstream `[{}] * n_itr` dictionary aliasing defect would repeat the final frame; the local provider creates independent dictionaries and preflights IDs/signatures. This is a calibration-data correctness fix, not a change to installed environments.
- Built an isolated CasADi-enabled TensorRT 10.7 runtime with Conda `modelopt_uniad_dl4agx` CUDA 11.8/nvcc/g++ and compared it with the accepted BFGS runtime on the same corrected INT8 engine for 200 frames. Planning CSV maximum absolute delta was `1e-7 m`; `avg L2=0.777632 m`, `box col=0.833333%`, and planning-to-PyTorch L2 `0.229213 m` were identical. The optimizer implementation is therefore not the remaining official parity defect, and the slower CasADi build is not promoted to the formal runtime.
- The remaining full-sequence FP32/FP16/INT8 planning gap is consequently classified as checkpoint/export/plugin/hardware numerical and temporal-contract variance after the calibration and post-processing audits; no additional malignant bug was reproduced in this round. NVIDIA's documented planning MSE label is retained as mean trajectory-point Euclidean L2, with coordinate MSE reported separately.
- CARLA inventory: 233 training route XMLs (`Town01 33`, `Town02 21`, `Town03 42`, `Town04 44`, `Town05 42`, `Town06 28`, `Town07 14`, `Town10HD 9`). Complete scenario JSONs contain 393,070 event configurations across Town01-Town06. These are event definitions, not 393,070 routes; the current custom driver executes one route at a time and does not implement the official ScenarioRunner/leaderboard scorer.
- Current full-route evidence remains one Town03 route for the INT8 service only. A four-backend x 233-route sweep for UniAD-tiny and UniV2X would be 1,864 serial route runs and is technically possible only after adding a CARLA route scheduler and scorer adapter; it has not been completed or represented as a full benchmark.

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

---

## Iteration 017 - 2026-08-14T15:29:17-07:00 (PDT)

### FP16 occupancy/Col 继续排查

- 所有新实验继续使用隔离的 `modelopt_uniad_dl4agx` 环境、Conda CUDA 11.8 工具链、TensorRT 10.9.0.34、当前 native TensorRT plugin 和 `official_literal` 时序协议；没有切换到系统 CUDA/G++，没有修改权重或数据集。
- 200 帧标准 FP16 baseline（profile `901/901/1600`）仍是 occupancy IoU `91.5980%` 左右；已有 6018 帧正式候选的 occupancy IoU/优化后 box Col 为 `71.278528% / 0.459732%`。本轮没有将任何候选提升为正式全序列结果。

### 新候选与结果

| 候选 | 构建/评估状态 | 200 帧 occupancy IoU | raw avg. L2 | raw box Col |
| --- | --- | ---: | ---: | ---: |
| FP16 full output-lineage FP32（旧候选） | 可构建，已有结果 | `92.3829%` | `0.721249 m` | `1.166667%` |
| FP16 full-lineage + PluginV2 FP32 | 构建失败 | - | - | - |
| FP16 full-lineage + layout FP32 | 构建失败 | - | - | - |
| FP16 仅 `Reshape_1957/1983/1997` FP32 | 可构建，但退化 | `90.3709%` | `0.738667 m` | `1.083333%` |
| FP16 官方 profile `901/901/1150` | 可构建，200 帧复核 | `91.2459%` | `0.737405 m` | `1.166667%` |

- `PluginV2.precision=FP32` 在 TensorRT 10.9 对 `MultiScaleDeformableAttnTRT_1998` 报 `No supported formats`，不能作为正式方案；强制大范围 Shuffle/Resize/Select precision 会触发 Myelin SSA/IR verifier 失败。两种失败都不是 OOM，也没有生成可用 engine。
- 仅保护三个 transformer 输入 Shuffle 能构建但 IoU 和 L2 退化，说明共享 transformer 数值路径不能用局部入口锁定解决。
- 将 profile 上界从 `1600` 改回官方 `1150` 没有改善，反而 200 帧 IoU 降至 `91.2459%`；因此 `1600` 只是早期固定容量诊断上界，不是 FP16 occupancy 根因。

### score dump 口径边界

- `seg_score_out` alias 在 TensorRT layer audit 中被融合到 `ForeignNode[Cast_24264...Unsqueeze_24263]`，内部输出为 Half 后再 reformat 为 Float。TensorRT API 已确认该 binding 与 `seg_out` 都声明为 `LINEAR`；但 `score > 0.1` 与同一 engine 的最终 `seg_out` 只保持正栅格总数一致，空间位置不逐元素一致。因此问题不是 C++ wrapper 漏处理非线性 stride，而是 Myelin 中间 alias 的物化结果不能作为最终 `Greater` 的语义等价输出。本轮不再用该 alias 做逐元素 occupancy 结论，正式门禁只使用 runtime 原生 `seg_out.packbits`。
- ONNX 图中实际阈值仍是 `Greater_24260(..., 0.1)`；现有阈值 sweep 和 6018 帧 score 审计已经证明全局阈值/形态学不是主因，不能通过调阈值伪造官方 Col。

### 结论

本轮排除了 profile 上界、插件单点、末端 Conv、三个入口 Shuffle、阈值以及评估统计口径。剩余差异位于每帧共享 image/BEV feature producer 到 occupancy consumer 的 FP16 融合数值传播；在 TensorRT 10.9 中直接锁定插件或大范围 layout 会破坏构建，局部锁定又退化。FP16 当前仍未通过 occupancy/Col 验收，正式 accuracy reference 继续使用 FP32，不能把本轮实验称为“FP16 已修复”。

---

## Iteration 018 - 2026-08-14T21:01:37-07:00 (PDT)

### 可靠 logits boundary

- Builder 新增默认关闭的 `--mark-output-direct-alias SOURCE=ALIAS`。它直接重命名并标记原始中间 tensor 为 network output，不插入会被 Myelin 继续融合的 Identity。
- FP16/FP32 direct-score engines 均使用官方 profile `901/901/1150`。两者在 200 帧中都满足 `seg_score_out > 0.1` 与各自 `seg_out` 的 `2,500,000/2,500,000` bit 完全一致，解决了旧 debug alias 不能逐元素解释的问题。
- 有效 FP16 threshold sweep 的最佳阈值为 `0.107`，occupancy IoU 仅从 `91.149326%` 提高到 `91.188137%`，再次证明调阈值不能修复 Col。

### FP16 与 FP32 有效 logits 对照

| 指标 | 200 帧结果 |
| --- | ---: |
| FP16 vs FP32 occupancy IoU | `91.849044%` |
| occupancy bit flips | `5,676` |
| score MAE | `0.001581457` |
| abs error p95 / p99 / p99.9 / max | `0.00042098 / 0.04099321 / 0.24271484 / 0.87691808` |
| horizon flips 0.5--2.5 s | `958 / 992 / 1143 / 1221 / 1362` |
| scene 首帧平均 flips | `1.0` |
| 非 scene 首帧平均 flips | `29.2268` |

- 关闭外部 temporal state 的诊断中，FP16/FP32 flips 从 `5,676` 降为 `749`，IoU 从 `91.849044%` 提高到 `95.263091%`，且五个 horizon flips 变为 `153/132/121/186/157`。这不是正式评估协议，但证明跨帧状态消费会放大 FP16 差异。
- 保护 `prev_bev` 的七个入口层（`Reshape_1547 -> ... -> Mul_1934`）可构建，但对 PyTorch IoU 为 `91.057226%`、对 FP32 IoU 为 `91.854624%`、仍有 `5,659` flips，基本没有改善。误差不在入口 reshape/rotate 单点，而在更深的 shared temporal encoder/BEV feature 传播。

### 当前验收状态

direct-output 修复的是逐层审计工具，不是正式 FP16 engine 精度。所有 precision 候选仍未通过 200 帧门禁，因此未重复 6018 帧长跑；正式 FP16 occupancy/Col 仍未验收，现有 6018 帧结果保持不变。

---

## Iteration 019 - 2026-08-14T21:29:22-07:00 (PDT)

### FP16 时序入口否证与中间特征二分

- 仅将 11 个 `prev_track_intances*` 直接消费 Mul 保持 FP32 后，对 PyTorch occupancy IoU 为 `90.629807%`，对 TRT FP32 IoU 为 `91.314118%`，翻转栅格由标准 FP16 的 `5,676` 增至 `6,030`；raw planning 为 `0.738576 m / 1.083333% box Col`。该候选退化，排除 track 输入消费单点。
- runtime 新增默认关闭的 `UNIAD_DUMP_BEV_EMBED` 和 `UNIAD_DUMP_AUDIT_OUTPUT`。Builder 的 direct alias 可将固定形状中间 tensor 命名为 `audit_out`；正式 engine 没有该输出且环境变量默认关闭，正式输出和时序协议不变。
- 新增 `UniAD/repro/scripts/compare_tensor_audit.py`，严格校验 manifest、文件字节数、帧数、shape 和 temporal protocol，再报告逐帧 MAE/RMSE/relative RMSE/cosine。

### FP16/FP32 中间张量结果

| 边界 | 帧数 | shape/frame | MAE mean | relative RMSE mean | cosine mean/min |
| --- | ---: | --- | ---: | ---: | ---: |
| recurrent `bev_embed` | 200 | `2500x1x256` | `0.00621349` | `1.489651%` | `0.999335 / 0.908129` |
| dense decoder `future_states.3` | 40 | `1x5x256x50x50` | `0.04092132` | `3.259410%` | `0.999070 / 0.986162` |

- `future_states.3` 五个未来 horizon 的 MAE 为 `0.033794 / 0.035245 / 0.038626 / 0.042905 / 0.054038`，随时域单调放大。40 帧同引擎 FP16/FP32 occupancy IoU 为 `95.729326%`，共 `1,148` flips。
- 该证据将根因边界从泛化的 shared temporal path 进一步收窄到 `bev_embed` 之后的 dense future feature/occupancy query 路径；末端 threshold、profile、prev_bev/track 入口和单个插件都已排除，但尚不能归因到唯一一个 TensorRT layer。

### tactic A/B 与验收结论

- 首次补测 builder `optimization_level=0` 的纯 FP16 engine；其 200 帧对 PyTorch occupancy IoU 只有 `90.116329%`，对 TRT FP32 为 `90.790114%`，共 `6,432` flips，raw/optimized planning 都为 `0.738535 m / 1.083333% box Col`。它比 level-3 标准 FP16 更差，排除 level-3 特定 tactic 是主因。
- 所有实验继续使用 `modelopt_uniad_dl4agx`、Conda CUDA 11.8、TensorRT `10.9.0.34` 和 official profile `901/901/1150`；固定 1150 仅用于已验证等价的中间张量诊断，未替换正式动态协议。
- 本轮修复并验证了可靠的逐层审计能力，但没有得到通过 200 帧门禁的 FP16 engine。故不启动新的 6018 帧长跑，不覆盖现有正式 FP16 表；FP32 仍是 accuracy reference，FP16 仍标记为 occupancy/Col 未验收。

---

## Iteration 020 - 2026-08-14T21:44:55-07:00 (PDT)

### Dense decoder 根因边界复核

- 新建 FP16/FP32 direct-output engine，在相同 200 帧、official-literal、固定 1150 诊断协议下导出 dense decoder 入口 `input.1743`。
- `input.1743` shape 为 `1x256x13x13`，FP16/FP32 MAE mean `0.00648140`、relative RMSE mean `0.870129%`、cosine mean/min `0.999598/0.938283`。同引擎最终 occupancy IoU 为 `91.606570%`、`5,831` flips。
- 与 `bev_embed` 的 relative RMSE `1.489651%` 和 decoder 输出 `future_states.3` 的 `3.259410%` 对照，入口误差没有放大；主要放大出现在 dense future decoder 内部，并随 horizon 从 MAE `0.033794` 增至 `0.054038`。

### 真正覆盖 decoder 的 FP32 候选

- 旧 `fp16_occ_shared96_fp32` 使用 `max_convolutions=0`；报告中的 Conv/Resize 只有 lineage 记录，`precision_constrained=false`，其 `90.393891%` IoU 不能代表 decoder 已被完整 FP32 化。
- 修正候选以 `future_states.3` 为根、depth `48`、最多 `5` 层卷积并启用 layout precision，TensorRT 报告确认 `21` 个 Conv、相关 Resize/Shuffle 以及合计 `376` 个数值层实际为 FP32；严格 `OBEY_PRECISION_CONSTRAINTS` 构建成功，engine SHA256 为 `baffbef44575601b55a957828da3fc7d957c8c8111df5a6d9779288205a61af4`。
- 该候选 200 帧对 PyTorch occupancy IoU 仅 `90.753124%`，对 TRT FP32 为 `91.448952%`，`5,934` flips；raw/optimized planning 均为 `0.738131 m / 1.083333% box Col`。model enqueue mean/p50/p99 为 `11.495/11.311/19.507 ms`。

### 结论

误差的可观测放大发生在 decoder，但强制该子图 FP32 会改变 TensorRT 融合边界并使 parity 退化，不能作为正式修复。至此已排除 profile、threshold、末端 Conv、插件、prev_bev/track 入口、builder level、decoder 不完整约束和 decoder 完整混合精度候选。没有新候选通过 200 帧门禁，不启动 6018 帧长跑；现有 FP16 全量结果与“未验收”标记保持不变。

---

## Iteration 022 - 2026-08-15T08:55:04-07:00 (PDT)

### FP16 MatMul/Mul 同源 A/B

- Builder 新增 `--force-fp32-operator MatMul|Mul`，使用 TensorRT layer type、FLOAT 输出门禁和严格 `OBEY_PRECISION_CONSTRAINTS`，不是名称列表传给 `trtexec` 的软提示。
- 同一修复后 ONNX SHA256 `82d0aa5e...`、同一标准插件 SHA256 `14ea7801...`、TensorRT `10.9.0.34`、optimization level 3、TF32 关闭，构建 baseline / MatMul-FP32 / Mul-FP32 / MatMul+Mul-FP32 四组。
- `MATRIX_MULTIPLY=557` 全部受保护，覆盖 MatMul 并保守包含 Gemm；ONNX 的 333 个 Mul 中 269 个 FLOAT Mul 全部受保护，剩余 49 INT64 + 15 INT32 为 shape/index 计算，不存在 FP16 量化。
- 官方 `901/901/1150` profile 的四组均在第 1225 帧 fail-closed：baseline 输出 1153 tracks，其余输出 1152，超过 1150。没有截断时序状态；另建同源 `901/901/1600` 四组并以 fixed 1600 完成全量。

完整 6018 帧、`official_literal`、同一输入和 collision optimizer：

| Engine | FP32 约束 | occupancy IoU | raw L2 / Col | planning MSE | optimized L2 / Col | candidates / modified frames | model / inference / E2E p50 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FP16 baseline | 无 | `71.329463%` | `0.729232 / 0.545585%` | `0.0618751` | `0.837594 / 0.440346%` | `635282 / 3425` | `11.247 / 15.178 / 67.772` |
| FP16 MatMul-FP32 | 557 MatrixMultiply | `71.337650%` | `0.728676 / 0.545585%` | `0.0624877` | `0.835565 / 0.429268%` | `631951 / 3417` | `13.324 / 17.228 / 70.473` |
| FP16 Mul-FP32 | 269 FLOAT Mul | `71.257946%` | `0.728846 / 0.545585%` | `0.0623445` | `0.835578 / 0.437576%` | `633874 / 3414` | `12.116 / 16.065 / 69.276` |
| FP16 MatMul+Mul-FP32 | 557 + 269 | `71.305393%` | `0.728650 / 0.548355%` | `0.0624150` | `0.835803 / 0.426498%` | `634138 / 3420` | `13.367 / 17.301 / 68.109` |

相对 baseline，MatMul-only occupancy IoU 仅 `+0.00819` 个百分点，Mul-only `-0.07152`，联合 `-0.02407`；最佳 Col 改善也只有 `-0.01385` 个百分点，仍远高于已验收 INT8 的约 `0.2631%`。同时 raw planning MSE 没有改善，MatMul 保护使 model p50 增加约 `18.47%`。这些结果否定“普通 FP16 未排除 MatMul/Mul 是 Col 异常主因”，但不否定个别融合边界对少数栅格的影响。

四组全量同时运行于 GPU 0--3；精度协议严格一致，延迟只用于本轮相对诊断，不替代串行正式延迟。TensorRT 的 LayerNorm FP16 warning 与既有 dense-future 误差放大证据仍然成立，下一步不应继续全局保护 MatMul/Mul。

证据：

- `UniAD/evidence/trained_tiny_fp16_matmul_mul_ab_full6018/summary_200.json`
- `UniAD/evidence/trained_tiny_fp16_matmul_mul_ab_full6018/summary_6018.json`
- 大文件：`/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20/fp16_matmul_mul_ab_20260815_max1600`

---

## 2026-08-15T23:28:46-07:00 (PDT) - seg_out 反向逐层首个显著误差定位

### 诊断协议

- 使用同一修正版 ONNX、同一标准插件、TensorRT `10.9.0.34`，分别构建 FP32/FP16 direct-output engine。
- 以同一批 40 帧记录输入逐个执行；关闭递归 temporal state 和 collision optimizer，使差异归因于 engine 数值路径，而不是两次运行的状态漂移。
- 从 `seg_out` 反向追踪到 occupancy decoder、dense future decoder、layer0 cross-attention，再追到 tracking score/active-index 分支。由于中间张量存在输入相关的动态 shape，runtime 改为保存 manifest v2 的 `shapes_per_frame`，不再用单一 shape 拒绝合法输出。
- 这是因果定位实验，不替代 official-literal 的 6018 帧递归验收。

### 首个因果边界

末端 occupancy `Mul`/`Sigmoid` 不是首个观测到的显著误差层。上游 tracking classification score 分支的 pre-Sigmoid tensor `onnx::Sigmoid_10611` 已出现全局 relative-RMSE `1.916788%`，最大绝对差 `5.026286`；经过 Sigmoid 的 `onnx::ReduceMax_10612` 为 `7.269288%`，随后固定容量的 `scores.1` 为 `4.713701%`。该分数随后参与 `0.35` 和 `0.4` 两个 active-index 条件。

实测阈值翻转示例：

| 帧 | 阈值 | index | FP32 | FP16 | 结果 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 19 | 0.35 | 824 | 0.3400408 | 0.3544922 | 翻转 |
| 19 | 0.40 | 681 | 0.4021551 | 0.3833008 | 翻转 |
| 21 | 0.40 | 561 | 0.4013114 | 0.3937988 | 翻转 |
| 28 | 0.40 | 396 | 0.4056823 | 0.3986816 | 翻转 |

随后出现离散 shape 分叉：

| 帧 | `track_scores` shape FP32 -> FP16 | 后续第一处可比大误差 |
| ---: | --- | --- |
| 19 | `[2] -> [1]` | cross-attention 输入长度改变 |
| 21 | `[3] -> [2]` | `input.1791` cross residual RRMS `58.016%` |
| 28 | `[2] -> [1]` | FFN hidden `input.1795` RRMS `32.187%` |

这说明首个因果边界是“tracking score 的 FP16 数值差异 + 0.35/0.4 离散筛选”，而不是 occupancy decoder 末端。对齐 shape 的帧中，occupancy gate `onnx::Less_23765` 的 `0.3` mask 没有发生翻转，因此此前怀疑的 occupancy gate threshold 不是这批异常帧的起点。

### 与 occupancy 路径的关系

从 `future_states.3` 到 `seg_out` 的 coarse 结果仍可复核：`future_states.3` relative-RMSE `10.9351%`，occupancy einsum `onnx::Slice_26434` 为 `0.3733%`，末端 `onnx::Mul_26440` 为 `2.8282%`。这些是后续误差或低能量张量的相对统计，不应倒推为最早根因。此前全图保护 557 个 MatrixMultiply 和 269 个 FLOAT Mul 的 A/B 结果也没有恢复 FP16 Col，因此不能把普通 occupancy-head MatMul/Mul Half 计算单独认定为主因。

这不表示 tracking score 分支内的 MatMul 已被排除；该分支本身包含 classification head MatMul/Gemm。当前已定位到第一个可观测连续边界是 `onnx::Sigmoid_10611`，下一步若继续修复，应只对 tracking classification score/active-index 分支做选择性 FP32 保护，并重新验证动态 track shape、occupancy 和 collision，而不是全局保护所有 MatMul/Mul。

### 复核文件

- 脚本：`UniAD/repro/scripts/run_fp16_seg_reverse_layer_audit.sh`
- 汇总：`UniAD/repro/scripts/summarize_fp16_seg_reverse_layer_audit.py`
- 轻量证据：`UniAD/evidence/trained_tiny_fp16_seg_reverse_layer_audit_20260815/summary.json`
- 原始 40 帧报告：`/home/lixingfeng/uniad_trt_artifacts/fp16_seg_reverse_20260815*/summary_*_40.json`

本轮没有覆盖正式 FP16 全量精度表，也没有宣称 FP16 已修复；结论是将异常从 occupancy 末端收窄到 tracking score 阈值分支，并证明 shape 分叉先于后续 occupancy/规划误差发生。

---

## 2026-08-16T00:42:08-07:00 (PDT) - MatMul 保护范围表述纠正

上一轮“tracking classification head 内部 MatMul 尚未逐个保护”的说法不准确。上一轮 `matmul_fp32` A/B engine 的 557 个 TensorRT `MATRIX_MULTIPLY` 已全部强制为 FP32，明确包含 tracking classification head 的 `MatMul_4474`、`MatMul_4504`、`MatMul_7804`；269 个 FLOAT `Mul` 也包含 `Mul_4500`、`Mul_7800`。所以单独再次保护 tracking MatMul/Mul 没有新的验证价值。

真正尚未隔离的是 score lineage 中的 `Div`、`Pow`、`Sub`、`Add`、`Relu`、`Sigmoid`、`ReduceMax` 和 TensorRT fusion 边界。后续若继续，应构建完整 tracking score lineage FP32 候选，或逐层保护这些非 MatMul 节点，不能把它描述为再次保护 tracking MatMul。

## Iteration 024 - 2026-08-16T01:45:24-07:00 (PDT) - Unified output storage and base result audit

### UniAD-tiny status

- The trained epoch-20 tiny deployment record is complete in this document. The formal 6018-frame TensorRT 10.9 section above includes the PyTorch reference, FP32, ordinary FP16, and accepted terminal-occupancy-protected INT8 rows with raw/optimized `avg. L2`, box collision, occupancy validity, modified-frame counts, and model/forward/E2E mean-p50-p99 latency.
- Tiny engines, ONNX graphs, calibration feeds, timing caches, frame predictions, and CARLA outputs are stored under `/data/lxf/uniad_deployment_outputs/trained_tiny_epoch20` and are not duplicated in the project source tree.

### UniAD-base full-validation result currently available

The base graph uses `base_e2e.py`/`base_e2e_trt_p.py`, the base 200x200 planning grid, fixed track capacity 1150, TensorRT `10.9.0.34`, and the same isolated Conda CUDA 11.8 toolchain. All four rows completed 6018 finite frames. The INT8 row is explicitly preliminary: its entropy calibration used only 8 training feeds (`calib8`), not the official tiny calibration protocol.

| Metric | PyTorch FP32 | TRT FP32 | TRT FP16 | TRT INT8(EQ)+FP16, calib8 |
| --- | ---: | ---: | ---: | ---: |
| frames | 6019 (6009 timed) | 6018 | 6018 | 6018 |
| planning avg. L2 | 0.913059 m | 2.559613 m | 2.408856 m | 2.380163 m |
| box Col | 0.066456% | 1.165947% | 1.024704% | 3.162734% |
| planning point-L2 vs PyTorch | 0 | 3.144457 m | 2.966040 m | 2.722219 m |
| forward mean / p50 / p99 | 517.867 / 470.277 / n/r ms | 215.580 / 214.395 / 233.181 ms | 137.403 / 135.917 / 162.725 ms | 126.623 / 125.679 / 143.087 ms |
| E2E mean / p50 / p99 | 586.327 / 545.332 / n/r ms | 368.480 / 364.289 / 440.199 ms | 289.559 / 284.672 / 379.506 ms | 276.915 / 273.425 / 349.483 ms |

`forward` is the synchronized TensorRT/PyTorch model call (`inference_call` for TensorRT); `p99` is unavailable for the historical PyTorch JSON because that runner stored only mean/p50/p95/max. The base engine runner currently emits planning and latency, not a complete detection/tracking/map NuScenes result object, so no base TRT mAP/AMOTA/NDS value is fabricated here. The base FP32 planning delta already exists before quantization; these rows are a completed runtime deployment audit, not an accuracy-parity acceptance.

### Generated-artifact cleanup and canonical locations

- Base generated outputs were moved to `/data/lxf/uniad_deployment_outputs/uniad_base_e2e`; the project path `uniad-trt/repro_20260806/artifacts/uniad_base_e2e` is now a symlink.
- Superseded random-weight `official_dummy`, incompatible `smoke_base_hybrid`, and the old root tiny `onnx` generated trees were removed from the local DL4AGX deployment artifacts. Training checkpoints under `stage1`/`stage2` were retained because they are weights, not stale ONNX/engine outputs.
- The old 30G UniV2X generated artifact tree under `UniV2X/deploy_int8_legacy_20260816/artifacts` was removed after verification that the accepted corrected outputs are already under `/data/lxf/univ2x_deployment_outputs/semantic_parity_20260810`. `AV-Solutions/univ2x-trt/artifacts` now links directly there; the old source compatibility directory remains only as a small source tree.
- No original `/data` dataset or checkpoint was modified. The canonical source root remains `/home/lixingfeng/UniAD_examine/DL4AGX`; generated model files are now kept on `/data` rather than consuming home storage.

---

## Iteration 025 - 2026-08-16T08:23:46-07:00 (PDT) - UniAD-base official-flow re-deployment

### Why the previous base table is superseded

The old TensorRT base rows (`FP32 avg. L2 2.559613 m`, `box Col 1.165947%`) are invalid as an accuracy-parity result. The base runtime had silently used the tiny RGB/ImageNet normalization `(x - mean) / std`. The base config instead requires decoded RGB JPEG pixels to be reordered to BGR and the BGR mean to be subtracted with unit standard deviation. After restoring a compile-time `UNIAD_IMAGE_NORM_MODE=1`, the first-frame normalized tensor differs from PyTorch/OpenCV by at most `2.38e-7`, and the first raw trajectory changed from the wrong `(-0.258119, 1.40344, ...)` to `(-0.326484, 3.13016, ...)`, close to PyTorch `(-0.325478, 3.133196, ...)`.

The base checkpoint also emits up to 1155 track rows during validation. The tiny tutorial's `MAX=1150` profile is therefore invalid for this checkpoint. The rebuilt base profile is `MIN/OPT/MAX=901/901/1400`; runtime buffers support 2200 rows, so this does not truncate temporal state.

### Official patch and export audit

- `uniad-onnx-export.patch` introduces `forward_uniad_trt` deployment paths across tracking, motion, occupancy, map/panseg and planning heads, exposes recursive track/BEV/timestamp/pose tensors, replaces unsupported decoding/control flow, and makes custom TRT operators exportable.
- `bevformer_tensorrt.patch` supplies the PyTorch-side symbolic/custom-op path for multi-scale deformable attention, DCNv2, rotate/inverse and related BEVFormer operations.
- `plugins-trt10-support.patch` ports the TensorRT plugins and kernels to the TRT 10 API; `uniad-torch1.12.patch`, `mmdet3d.patch` and `nuscenes-devkit.patch` align the framework, dataset and evaluation stack; `uniad-tiny-training-support.patch` is training/config support and does not itself quantize the graph.
- Export loads the matching checkpoint/config, executes six recursive `forward_uniad_trt` calls before `torch.onnx.export`, exports opset 16 with custom operators, then changes every ONNX `Reshape.allowzero` to `1`. The fresh repaired base graph is deterministic and byte-identical to the previous repaired graph: SHA256 `b1da46ab05127ae19c1e324f08ab0bfad3c679b91c725f826a071ecdfb2aad4f`. This isolates the earlier FP32 failure to runtime preprocessing/protocol rather than ONNX nondeterminism.

### Calibration semantics and 4090 compatibility

The repaired calibration script preserves the official temporal meaning: external track/BEV/timestamp state is initialized only at global sample 0; a scene boundary sets `use_prev_bev=0` while the external previous state is still carried; every exact `prev_track_intances0.shape[0] == 901` feed is stored as an independent 24-input dictionary. It fixes only the official script's loop-overwrite and counter-reset save bugs and writes a manifest/feed hash audit.

The first 10% train scan (2813 frames) produced 11 unique feeds: sample 0 plus samples `1600-1605` and `1633-1636`. Provider validation reports 11 independent dictionary identities and 11 unique 24-input signatures. Because those natural hits are highly clustered, they are retained as a pipeline/memory gate rather than accepted as the final INT8 calibration set; a separate 15% train scan is in progress.

ModelOpt 0.29 originally configures an 80 GiB TensorRT EP workspace, which caused augmented base calibration to exhaust a 24 GB RTX 4090. The project wrapper now bounds calibration TRT workspace to 4 GiB, disables auxiliary streams and uses builder level 0 without modifying the installed Conda environment or ModelOpt source. The required official EP order (`trt`, `cuda:0`, `cpu`), entropy calibration, `MatMul` exclusion, `dq_only`, graph simplification and custom plugins remain enabled.

### Reproducibility controls added

- Base engine build and runtime now share the same `build_base_recheck/libuniad_plugin.so`; build/profile suffixes prevent an unverified candidate from overwriting the accepted engine.
- Base evaluation defaults to `official_literal`, explicitly exports `UNIAD_COLLISION_OPTIMIZATION=1`, records compressed occupancy, and writes model/inference/E2E mean-p50-p99 plus collision-optimizer audit counts.
- A dedicated PyTorch `forward_uniad_trt` validation runner writes raw trajectories and a checkpoint/protocol manifest, so `planning MSE` can be recomputed against an exact protocol-matched reference rather than the older unmanifested Python test CSV.
- Generated ONNX/engines/evaluation remain under `/data/lxf/uniad_deployment_outputs/uniad_base_e2e`; duplicate and invalid intermediate ONNX/FP32 debug engines were removed, recovering about 6 GB. Dataset and checkpoint files were not modified.

---

## Iteration 026 - 2026-08-16T10:03:30-07:00 (PDT) - Base calibration memory gate and RotateTRT creator repair

### Workspace and storage boundaries

`TensorRT workspace` is a temporary GPU-memory budget used while TensorRT searches tactics and executes the ORT calibration subgraph. It is unrelated to the shell working directory or filesystem capacity. The calibration process cwd was `/home/lixingfeng/UniAD_examine`; the active deployment worktree is `DL4AGX/AV-Solutions/uniad-trt/repro_20260806`, while generated ONNX, calibration, engine and evaluation artifacts remain under `/data/lxf/uniad_deployment_outputs/uniad_base_e2e`.

The earlier statement that an observed 2.6 GiB process footprint proved the 80 GiB workspace issue closed was incorrect: that sample was taken before the augmented graph executed. A 4 GiB tactic workspace alone still failed on the monolithic 662-output calibration graph while requesting another 71,270,400-byte buffer. The accepted mechanism combines:

- TensorRT EP workspace `4 GiB`, auxiliary streams `0`, builder level `0`;
- entropy calibration tensors split into 11 chunks, maximum 64 tensors per chunk;
- each of the 11 independent feed dictionaries rewound and replayed for every chunk;
- shared Q/DQ tensor components kept in the same chunk;
- each child ORT/TensorRT session released before the next chunk.

The real base gate completed all 11 chunks and all 662 tensors. Observed GPU usage varied by activation shape and reached about 15.5 GiB in the heaviest sampled chunk, then returned to about 2--4 GiB between chunks. No CUDA allocation failure recurred. The 11 feeds remain a memory/mechanism gate, not a final representative calibration set: scanning the first 15% train prefix still produced the same clustered 11 exact-901 feeds.

Evidence:

- chunk audit: `/data/lxf/uniad_deployment_outputs/uniad_base_e2e/audits/base_train10pct_entropy_chunks_20260816_v3.json` (`status=complete`, chunks `0..10`);
- gate Q/DQ ONNX: `/data/lxf/uniad_deployment_outputs/uniad_base_e2e/onnx/recheck_20260816/uniad_base_e2e_int8_eq_train10pct_chunked_gate_v3.onnx`, SHA256 `47bba5f6befe5b8dc16fba3f116a4a9070c7ea93b09aced158ce6554a70e9e01`;
- quantization audit: `/data/lxf/uniad_deployment_outputs/uniad_base_e2e/audits/base_train10pct_chunked_gate_v3_quantization_inspection.json`;
- TensorRT 10.9 parser: `parsed=True`, `errors=0`, 24 inputs, 23 outputs, 33,850 layers.

The gate graph contains 602 quantized nodes. Validation confirms explicit DQ nodes, INT8 initializers, finite positive scales, 11/11 unique calibration feeds, and zero MatMul nodes consuming INT8 weight initializers. ModelOpt may still report MatMul activation Q/DQ propagation; that is distinct from quantizing MatMul weights.

### RotateTRT `center` warning

The ONNX symbolic exports `RotateTRT(img, angle, center, interpolation_i=...)`; `center` is the third runtime tensor input. `RotatePlugin::configurePlugin()` also requires `nbInputs == 3`. The old plugin creator nevertheless advertised both `interpolation` and `center` as build-time `PluginField` attributes, while `createPlugin()` only reads `interpolation`. TensorRT 10.9 therefore warned that the ONNX node had no `center` attribute even though the runtime input was present.

The fix removes only the erroneous creator field declaration from both RotateTRT creators and retains the three-input runtime ABI and kernel. The plugin was rebuilt with the isolated `modelopt_uniad_dl4agx` Conda CUDA 11.8/GCC toolchain and TensorRT `10.9.0.34`. The repaired FP and Q/DQ graphs parse without the warning. This metadata defect was not the calibration OOM cause and did not require changing the rotate kernel.

### Protocol-matched base FP32/FP16 full-sequence check

Both engines ran 6018 frames with `official_literal`, fixed track capacity 1400, collision optimization enabled, and the exact new PyTorch `forward_uniad_trt` raw trajectory reference. `mean(dx^2+dy^2)` is now emitted separately from mean point L2; the historical `planning_mse` alias is retained for compatibility and must not be read as a squared error.

| Engine/output | avg. L2 | box Col | mean point L2 vs PyTorch | mean(dx^2+dy^2) |
| --- | ---: | ---: | ---: | ---: |
| PyTorch raw deployment reference | 0.817744 m | 0.600975% | 0 | 0 |
| TRT FP32 raw | 0.804913 m | 0.653595% | 0.115536 m | 0.603189 m^2 |
| TRT FP32 collision-optimized | 0.847111 m | 0.207710% | 0.196022 m | 0.666626 m^2 |
| TRT FP16 raw | 0.816562 m | 0.708984% | 0.539206 m | 1.193325 m^2 |
| TRT FP16 collision-optimized | 0.858995 m | 0.282486% | 0.596742 m | 1.258518 m^2 |

| Engine | model enqueue mean/p50/p99 | synchronized inference call mean/p50/p99 | E2E mean/p50/p99 |
| --- | ---: | ---: | ---: |
| TRT FP32 | 204.698 / 204.115 / 210.275 ms | 233.194 / 231.043 / 261.908 ms | 344.725 / 339.708 / 437.793 ms |
| TRT FP16 | 106.361 / 105.706 / 111.661 ms | 133.848 / 132.541 / 147.891 ms | 245.114 / 242.500 / 311.807 ms |

These rows supersede the old wrong-normalization base deployment table for FP32/FP16. They do not yet close the final base INT8 acceptance because the current successful Q/DQ artifact used only the 11-feed mechanism gate and has not been promoted to a representative calibration engine/full-sequence result.
