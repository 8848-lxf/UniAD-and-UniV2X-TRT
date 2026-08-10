# UniAD TensorRT/INT8 4090 Handoff

Timeline timestamps use the active process timezone (PDT, UTC-07:00).

Status is fail-closed: a smoke test is not counted as full validation, and the official random-weight ONNX results are not treated as model-accuracy evidence.

## Local lineage

- Working root: `/home/lixingfeng/UniAD_examine/DL4AGX/AV-Solutions/uniad-trt/repro_20260806`
- Original checkpoint: `/home/lixingfeng/data/ckpts/uniad_base_e2e.pth`
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
| UniAD-base config and checkpoint adaptation | Completed | Uses the base graph, base input metadata, and dynamic temporal-track profile; the checkpoint is not inserted into the tiny graph. |
| Base FP32 ONNX export | Completed | Local artifact excluded from Git. |
| Base explicit-QDQ INT8 graph | Completed | ONNX check passes with expected TRT plugin-domain handling; MatMul weights/activations are excluded from INT8. |
| Base FP32/FP16/INT8 engine build | Completed | Local `.engine` files excluded from Git. |
| Base engine finite-output smoke | Completed | `UniAD/evidence/uniad_base_e2e/base_engine_smoke_sample0.json`. |
| Base PyTorch full 6018-frame evaluation | Completed | Original checkpoint baseline recorded below. |
| Base TensorRT FP32 full 6018-frame evaluation | Running at snapshot time | Progress was 4865/6018; do not report final metrics until the process exits and output files are complete. |
| Base TensorRT FP16 full evaluation | Pending | Run after FP32. |
| Base TensorRT INT8 full evaluation | Pending | Run after FP16; current base calibration set contains only 8 training samples and must be called out in accuracy interpretation. |
| CARLA closed-loop evaluation | Not completed | Existing CARLA assets are partial/reused; the full download was explicitly paused. |

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

## Resume procedure

1. Confirm GPU 6 is still reserved and inspect the FP32 process/log before starting anything else.
2. Finish FP32 and verify its summary, frame count, finite trajectory count, and mean/p50/p99 files.
3. Run FP16 and INT8 with the same dataset ordering, warmup, synchronization, and end-to-end definition.
4. Compare planning output against PyTorch and separately state which full detection/tracking/map metrics the engine runner actually reconstructs.
5. Increase/rebuild base INT8 calibration if full-validation degradation is excessive; do not silently compare an 8-sample calibration against a larger protocol.
6. Update this document and push one commit after each completed major round.

---

## Iteration 001 - 2026-08-09T18:33:26-07:00

- Created the publication branch and source-only repository layout.
- Imported the UniAD export/quantization/evaluation scripts, deployment model changes, enqueueV3 runtime source, and small audit reports.
- Recorded the original-checkpoint PyTorch baseline, base engine smoke trend, active FP32 full evaluation, and remaining full-validation/CARLA work.
- Excluded weights, ONNX graphs, engines, datasets, calibration tensors, timing caches, compiled libraries, and large generated outputs.
