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


class ImageEncoder(torch.nn.Module):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def forward(self, image):
        return tuple(self.agent.extract_img_feat(image))


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
    errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not parsed or errors:
        raise RuntimeError(errors)
    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 * 1024 ** 3)
    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - start
    if serialized is None:
        raise RuntimeError("Image encoder engine build failed")
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
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    _, agent = build_trt_agent(args.config, args.checkpoint, args.agent)
    model = ImageEncoder(agent).cuda().eval()
    image = load_export_inputs(args.input_npz, torch.device("cuda"))[25]
    with torch.no_grad():
        reference = tuple(value.detach().cpu() for value in model(image))

    output_names = ["feature_%d" % index for index in range(len(reference))]
    raw_onnx = os.path.join(args.output_dir, "image_encoder_raw.onnx")
    deploy_onnx = os.path.join(args.output_dir, "image_encoder_deploy.onnx")
    engine_path = os.path.join(args.output_dir, "image_encoder_fp32.engine")
    torch.onnx.export(
        model,
        (image,),
        raw_onnx,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        input_names=["img"],
        output_names=output_names,
        opset_version=16,
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    )
    graph = onnx.load(raw_onnx, load_external_data=True)
    rewritten = 0
    for node in graph.graph.node:
        if node.domain == "mmcv" and node.op_type == "MMCVModulatedDeformConv2d":
            node.domain = ""
            node.op_type = "ModulatedDeformableConv2dTRT"
            rewritten += 1
    onnx.save_model(
        graph,
        deploy_onnx,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="image_encoder_deploy.onnx.data",
        size_threshold=1024,
    )
    del graph, model, agent, image
    gc.collect()
    torch.cuda.empty_cache()

    build_seconds, layers = build_engine(deploy_onnx, engine_path, args.plugin)
    runtime = TensorRTEngine(engine_path, args.plugin)
    image = load_export_inputs(args.input_npz, torch.device("cuda"))[25]
    actual = runtime.infer({"img": image})
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
        "network_layers": layers,
        "build_seconds": build_seconds,
        "outputs": outputs,
    }
    with open(os.path.join(args.output_dir, "comparison.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
