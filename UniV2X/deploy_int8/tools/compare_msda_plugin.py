import argparse
import ctypes
import importlib.util
import json
import os
import time

import tensorrt as trt
import torch

from trt_engine import TensorRTEngine


class MultiScaleDeformableAttentionModel(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, value, spatial_shapes, reference, offsets, weights):
        return self.function(value, spatial_shapes, reference, offsets, weights)


def load_function(path):
    spec = importlib.util.spec_from_file_location("univ2x_msda_function", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.multi_scale_deformable_attn


def build_engine(onnx_path, engine_path, plugin_path, workspace_gib):
    ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
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
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_gib * 1024 ** 3
    )
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError("MSDA TensorRT engine build failed")
    with open(engine_path, "wb") as handle:
        handle.write(serialized)
    return build_seconds, network.num_layers


def synchronize_latency(function, warmup, iterations):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        stop.record()
        stop.synchronize()
        values.append(start.elapsed_time(stop))
    values = torch.tensor(values, dtype=torch.float64)
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(torch.quantile(values, 0.50)),
        "p99_ms": float(torch.quantile(values, 0.99)),
    }


def tensor_quantile(value, quantile):
    flattened = value.flatten()
    rank = max(1, min(flattened.numel(), int(quantile * flattened.numel())))
    return flattened.kthvalue(rank).values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--function", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--query-count", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--workspace-gib", type=int, default=2)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(0)
    device = torch.device("cuda")
    spatial_shapes = torch.tensor(
        [[136, 240], [68, 120], [34, 60], [17, 30]],
        dtype=torch.int64,
        device=device,
    )
    spatial_size = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum())
    inputs = (
        torch.randn(args.batch_size, spatial_size, 8, 32, device=device),
        spatial_shapes,
        torch.rand(args.batch_size, args.query_count, 1, 8, device=device),
        torch.randn(args.batch_size, args.query_count, 8, 64, device=device),
        torch.randn(args.batch_size, args.query_count, 8, 32, device=device),
    )
    input_names = ("value", "spatial_shapes", "reference", "offsets", "weights")
    model = MultiScaleDeformableAttentionModel(load_function(args.function)).cuda().eval()

    with torch.no_grad():
        reference = model(*inputs).detach()
    onnx_path = os.path.join(args.output_dir, "msda_real_shape.onnx")
    engine_path = os.path.join(args.output_dir, "msda_real_shape.engine")
    torch.onnx.export(
        model,
        inputs,
        onnx_path,
        input_names=input_names,
        output_names=["output"],
        opset_version=16,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    )
    build_seconds, network_layers = build_engine(
        onnx_path, engine_path, args.plugin, args.workspace_gib
    )
    runtime = TensorRTEngine(engine_path, args.plugin)
    input_dict = dict(zip(input_names, inputs))
    actual = runtime.infer(input_dict)["output"]
    difference = (reference.float() - actual.float()).abs()
    reference_abs = reference.float().abs()

    with torch.no_grad():
        pytorch_latency = synchronize_latency(
            lambda: model(*inputs), args.warmup, args.iterations
        )
        trt_latency = synchronize_latency(
            lambda: runtime.infer(input_dict, synchronize=False),
            args.warmup,
            args.iterations,
        )
    report = {
        "plugin": os.path.abspath(args.plugin),
        "shape_contract": {
            "batch_size": args.batch_size,
            "query_count": args.query_count,
            "spatial_shapes": spatial_shapes.cpu().tolist(),
            "spatial_size": spatial_size,
            "output_shape": list(reference.shape),
        },
        "build": {
            "seconds": build_seconds,
            "network_layers": network_layers,
            "tf32": False,
            "workspace_gib": args.workspace_gib,
        },
        "numerics": {
            "mean_abs_error": float(difference.mean()),
            "p50_abs_error": float(tensor_quantile(difference, 0.50)),
            "p99_abs_error": float(tensor_quantile(difference, 0.99)),
            "max_abs_error": float(difference.max()),
            "reference_mean_abs": float(reference_abs.mean()),
            "allclose_rtol_1e-3_atol_1e-4": bool(
                torch.allclose(reference.float(), actual.float(), rtol=1.0e-3, atol=1.0e-4)
            ),
        },
        "latency": {
            "pytorch_mmcv_cuda": pytorch_latency,
            "tensorrt_plugin": trt_latency,
        },
    }
    output_path = os.path.join(args.output_dir, "comparison.json")
    with open(output_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
