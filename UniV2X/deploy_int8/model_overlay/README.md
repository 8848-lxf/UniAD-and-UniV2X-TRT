# UniV2X TensorRT model overlay

This directory contains the model-side changes layered on the NVIDIA UniAD
TensorRT deployment tree. It intentionally does not duplicate the complete
upstream UniAD or `third_party/uniad_mmdet3d` source.

Apply it to a prepared UniAD deployment checkout before running the tools:

```bash
cp -a UniV2X/deploy_int8/model_overlay/projects/. \
  /path/to/UniV2X_deploy/projects/
```

The destination must already contain the NVIDIA ONNX-export patch stack and
the borrowed `third_party/uniad_mmdet3d` package described by the upstream
`uniad-trt` setup documentation. Set `PYTHONPATH` so the prepared deployment
tree and its `projects/mmdet3d_plugin/uniad/functions` directory precede the
original UniV2X checkout.

The overlay includes the two-agent TensorRT wrapper, cooperative track/map/BEV
fusion, temporal-state parity fixes, motion/occupancy masking, and raw map
candidate outputs used for data-dependent host postprocessing. The raw map
path replaces the non-generalizable Python control flow that ONNX tracing
would otherwise freeze into `lane_pred`.

After plugin repair, run `tools/promote_map_position_input_onnx.py` on each
agent graph before engine construction. It replaces the traced static map
position branch with the explicit `[1, 40000, 256]` input supplied and cached
by `trt_engine.py`; this avoids graph-internal Myelin configuration rebuilds
without changing the model's map semantics.
