import argparse
import ctypes
import gc
import json
import os
import time

import onnx
import tensorrt as trt
import torch

import projects.mmdet3d_plugin  # noqa: F401

from trt_engine import TensorRTEngine
from trt_runtime import build_trt_agent, load_export_inputs


class BevEncoderLayers(torch.nn.Module):
    def __init__(self, agent, attention_internals=False):
        super().__init__()
        self.agent = agent
        self.agent.pts_bbox_head.transformer.encoder.return_intermediate = True
        self.stage_outputs = []
        self.stage_names = []
        first_layer = self.agent.pts_bbox_head.transformer.encoder.layers[0]
        if os.environ.get("UNIV2X_TRT_NATIVE_MSDA") == "1":
            first_layer.attentions[1].deformable_attention.use_native_grid_sample = True
        for name, module in (
            ("layer0_temporal_attention", first_layer.attentions[0]),
            ("layer0_norm0", first_layer.norms[0]),
            ("layer0_spatial_attention", first_layer.attentions[1]),
            ("layer0_norm1", first_layer.norms[1]),
            ("layer0_ffn", first_layer.ffns[0]),
            ("layer0_norm2", first_layer.norms[2]),
        ):
            module.register_forward_hook(
                lambda module, inputs, output, stage_name=name:
                self._capture_stage(stage_name, output)
            )
        if attention_internals:
            spatial_attention = first_layer.attentions[1]
            deformable_attention = spatial_attention.deformable_attention
            for name, module in (
                ("layer0_spatial_value_proj", deformable_attention.value_proj),
                ("layer0_spatial_sampling_offsets", deformable_attention.sampling_offsets),
                ("layer0_spatial_attention_logits", deformable_attention.attention_weights),
                ("layer0_spatial_output_proj", spatial_attention.output_proj),
            ):
                module.register_forward_hook(
                    lambda module, inputs, output, stage_name=name:
                    self._capture_stage(stage_name, output)
                )
            original_msda = deformable_attention.multi_scale_deformable_attn

            def capture_msda(*inputs):
                output = original_msda(*inputs)
                self._capture_stage("layer0_spatial_msda", output)
                return output

            deformable_attention.multi_scale_deformable_attn = capture_msda

    def _capture_stage(self, name, output):
        self.stage_names.append(name)
        self.stage_outputs.append(output)

    def forward(
        self, image, can_bus, lidar2img, image_shape, prev_bev, use_prev_bev
    ):
        self.stage_outputs = []
        self.stage_names = []
        image_features = self.agent.extract_img_feat(image)
        layers, _ = self.agent.pts_bbox_head.get_bev_features_trt(
            image_features,
            can_bus,
            lidar2img,
            image_shape,
            prev_bev,
            use_prev_bev,
        )
        stages = tuple(
            stage.permute(1, 0, 2) if stage.shape[0] == 1 else stage
            for stage in self.stage_outputs
        )
        return stages + tuple(layer.permute(1, 0, 2) for layer in layers)


def build_engine(onnx_path, engine_path, plugin, workspace_gib, precision):
    ctypes.CDLL(plugin, mode=ctypes.RTLD_GLOBAL)
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as handle:
        parsed = parser.parse(handle.read(), onnx_path)
    errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
    if not parsed or errors:
        raise RuntimeError(errors)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gib * 1024 ** 3
    )
    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - start
    if serialized is None:
        raise RuntimeError("BEV encoder engine build failed")
    with open(engine_path, "wb") as handle:
        handle.write(serialized)
    return build_seconds, network.num_layers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("input_npz")
    parser.add_argument("plugin")
    parser.add_argument("--agent", choices=("ego", "infrastructure"), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workspace-gib", type=int, default=8)
    parser.add_argument("--attention-internals", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    _, agent = build_trt_agent(args.config, args.checkpoint, args.agent)
    model = BevEncoderLayers(
        agent, attention_internals=args.attention_internals
    ).cuda().eval()
    values = load_export_inputs(args.input_npz, torch.device("cuda"))
    inputs = (values[25], values[26], values[27], values[28], values[17], values[30])
    input_names = (
        "img", "can_bus", "lidar2img", "image_shape", "prev_bev", "use_prev_bev"
    )
    with torch.no_grad():
        reference = tuple(value.detach().cpu() for value in model(*inputs))
    output_names = tuple(model.stage_names) + tuple(
        "encoder_layer_%d" % index for index in range(6)
    )

    raw_onnx = os.path.join(args.output_dir, "bev_encoder_layers_raw.onnx")
    deploy_onnx = os.path.join(args.output_dir, "bev_encoder_layers_deploy.onnx")
    engine_path = os.path.join(
        args.output_dir, "bev_encoder_layers_%s.engine" % args.precision
    )
    torch.onnx.export(
        model,
        inputs,
        raw_onnx,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=16,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    )
    graph = onnx.load(raw_onnx, load_external_data=True)
    rewritten = 0
    scatter_add_reductions = []
    for node in graph.graph.node:
        if node.domain == "mmcv" and node.op_type == "MMCVModulatedDeformConv2d":
            node.domain = ""
            node.op_type = "ModulatedDeformableConv2dTRT"
            rewritten += 1
        if node.op_type == "ScatterElements" and not any(
            attribute.name == "reduction" for attribute in node.attribute
        ):
            node.attribute.append(onnx.helper.make_attribute("reduction", "add"))
            scatter_add_reductions.append(node.name)
    onnx.save_model(
        graph,
        deploy_onnx,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="bev_encoder_layers_deploy.onnx.data",
        size_threshold=1024,
    )
    del graph, model, agent, values, inputs
    gc.collect()
    torch.cuda.empty_cache()

    build_seconds, layers = build_engine(
        deploy_onnx, engine_path, args.plugin, args.workspace_gib, args.precision
    )
    runtime = TensorRTEngine(engine_path, args.plugin)
    values = load_export_inputs(args.input_npz, torch.device("cuda"))
    inputs = (values[25], values[26], values[27], values[28], values[17], values[30])
    actual = runtime.infer(dict(zip(input_names, inputs)))

    outputs = {}
    for name, expected in zip(output_names, reference):
        observed = actual[name].cpu()
        difference = (expected.float() - observed.float()).abs()
        outputs[name] = {
            "shape": list(expected.shape),
            "mean_abs_error": float(difference.mean()),
            "max_abs_error": float(difference.max()),
            "allclose_rtol_1e-3_atol_1e-4": bool(torch.allclose(
                expected.float(), observed.float(), rtol=1.0e-3, atol=1.0e-4
            )),
        }
    report = {
        "agent": args.agent,
        "rewritten_deformable_convolutions": rewritten,
        "scatter_add_reductions": scatter_add_reductions,
        "network_layers": layers,
        "workspace_gib": args.workspace_gib,
        "precision": args.precision,
        "attention_internals": args.attention_internals,
        "build_seconds": build_seconds,
        "outputs": outputs,
    }
    with open(os.path.join(args.output_dir, "comparison.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
