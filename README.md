# DL4AGX

[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-yellow.svg)](https://conventionalcommits.org)

This repository contains model designs, deployment solutions for state-of-the-art networks, and inference samples for Autonomous Vehicle (AV) applications on NVIDIA DRIVE Platforms.

## Table of Contents
- [Deployment and Inference Solutions](./AV-Solutions/)
   - [ONNX Export Guidance for TensorRT](./AV-Solutions/onnx-export-guidance/)
   - [BEVFormer INT8 Explicit Quantization](./AV-Solutions/bevformer-int8-eq/)
   - [DCNv4 TensorRT](./AV-Solutions/dcnv4-trt/)
   - [Far3D TensorRT](./AV-Solutions/far3d-trt/)
   - [Deploy LLMs with TensorRT-LLM](./AV-Solutions/llms-trtllm)
   - [MTMI TensorRT](./AV-Solutions/mtmi/)
   - [PETRv1&v2 TensorRT](./AV-Solutions/petr-trt/)
   - [Sparsity INT8 Training and TensorRT Inference](./AV-Solutions/SparsityINT8/)
   - [StreamPETR TensorRT](./AV-Solutions/streampetr-trt/)
   - [UniAD TensorRT](./AV-Solutions/uniad-trt/)
   - [VAD-TensorRT](./AV-Solutions/vad-trt/)
   - [UniV2X TensorRT](./AV-Solutions/univ2x-trt/)

## Local unified deployment handoff

For the RTX 4090 continuation of the UniAD, VAD-Tiny, and UniV2X deployment
chains, use `/home/lixingfeng/UniAD_examine/DL4AGX` as the working root.
The canonical PDT-timestamped development records are in
[`AV-Solutions/docs/`](./AV-Solutions/docs/). This local handoff is maintained
on the `feature/dl4agx-unified-4090` branch of the companion deployment
repository. Generated engines, ONNX files, checkpoints, calibration arrays,
datasets, and build directories stay local and are excluded from Git.
   
- [Hardware-friendly Models](./Models/)
   - [DEST](./Models/DEST/)
   - [ReduceFormer](./Models/ReduceFormer/)
   - [Swin-Free](./Models/SwinFree/)
