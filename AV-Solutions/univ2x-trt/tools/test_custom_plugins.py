import argparse
import ctypes
import importlib.util
import json
import os
import tempfile

import tensorrt as trt
import torch

from trt_engine import TensorRTEngine


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ModulatedDeformConvModel(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, input_value, offset, mask, weight):
        return self.function(
            input_value, offset, mask, weight, None,
            stride=1, padding=1, dilation=1, groups=1, deform_groups=1,
        )


class MultiScaleDeformableAttentionModel(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, value, spatial_shapes, reference, offsets, weights):
        return self.function(value, spatial_shapes, reference, offsets, weights)


class RotateModel(torch.nn.Module):
    def __init__(self, function):
        super().__init__()
        self.function = function

    def forward(self, image, angle, center):
        return self.function(image, angle, center, interpolation="bilinear")


def build_engine(onnx_path, engine_path, plugin):
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
    if not parsed or parser.num_errors:
        raise RuntimeError([str(parser.get_error(i)) for i in range(parser.num_errors)])
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 ** 3)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("Plugin test engine build failed")
    with open(engine_path, "wb") as handle:
        handle.write(serialized)


def run_case(name, model, inputs, input_names, plugin, directory):
    model = model.cuda().eval()
    inputs = tuple(value.cuda() for value in inputs)
    with torch.no_grad():
        reference = model(*inputs).detach()
    onnx_path = os.path.join(directory, name + ".onnx")
    engine_path = os.path.join(directory, name + ".engine")
    torch.onnx.export(
        model,
        inputs,
        onnx_path,
        input_names=input_names,
        output_names=["output"],
        opset_version=16,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    )
    build_engine(onnx_path, engine_path, plugin)
    engine = TensorRTEngine(engine_path, plugin)
    actual = engine.infer(dict(zip(input_names, inputs)))["output"]
    difference = (reference.float() - actual.float()).abs()
    return {
        "shape": list(reference.shape),
        "mean_abs_error": float(difference.mean()),
        "max_abs_error": float(difference.max()),
        "allclose_rtol_1e-3_atol_1e-4": bool(torch.allclose(
            reference.float(), actual.float(), rtol=1.0e-3, atol=1.0e-4
        )),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions-dir", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    deform = load_module(
        "univ2x_deform_function",
        os.path.join(args.functions_dir, "modulated_deformable_conv2d.py"),
    )
    attention = load_module(
        "univ2x_msda_function",
        os.path.join(args.functions_dir, "multi_scale_deformable_attn.py"),
    )
    rotate = load_module(
        "univ2x_rotate_function",
        os.path.join(args.functions_dir, "rotate.py"),
    )
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory(prefix="univ2x_plugin_test_") as directory:
        deform_inputs = (
            torch.randn(1, 8, 16, 16),
            torch.randn(1, 18, 16, 16),
            torch.sigmoid(torch.randn(1, 9, 16, 16)),
            torch.randn(8, 8, 3, 3),
        )
        deform_result = run_case(
            "deform",
            ModulatedDeformConvModel(deform.modulated_deformable_conv2d),
            deform_inputs,
            ["input", "offset", "mask", "weight"],
            args.plugin,
            directory,
        )

        spatial_shapes = torch.tensor(
            [[8, 12], [4, 6], [2, 3], [1, 2]], dtype=torch.int64
        )
        spatial_size = int((spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum())
        query_count = 50
        attention_inputs = (
            torch.randn(1, spatial_size, 8, 32),
            spatial_shapes,
            torch.rand(1, query_count, 1, 8),
            torch.randn(1, query_count, 8, 64),
            torch.randn(1, query_count, 8, 32),
        )
        attention_result = run_case(
            "attention",
            MultiScaleDeformableAttentionModel(
                attention.multi_scale_deformable_attn
            ),
            attention_inputs,
            ["value", "spatial_shapes", "reference", "offsets", "weights"],
            args.plugin,
            directory,
        )

        rotate_result = run_case(
            "rotate",
            RotateModel(rotate.rotate),
            (
                torch.randn(256, 200, 200),
                torch.tensor([37.0]),
                torch.tensor([100.0, 100.0]),
            ),
            ["image", "angle", "center"],
            args.plugin,
            directory,
        )

    result = {
        "plugin": os.path.abspath(args.plugin),
        "modulated_deformable_conv2d": deform_result,
        "multi_scale_deformable_attention": attention_result,
        "rotate": rotate_result,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
