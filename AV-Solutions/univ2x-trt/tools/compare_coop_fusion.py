import argparse
import ctypes
import json
import os
import time

import numpy as np
import onnx
import tensorrt as trt
import torch

import projects.mmdet3d_plugin  # noqa: F401

from trt_engine import TensorRTEngine
from trt_runtime import build_trt_agent, empty_track_state


class AgentFusion(torch.nn.Module):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def forward(self, *inputs):
        inf_tracks = inputs[:14]
        veh_tracks = inputs[14:28]
        match_vehicle_index = inputs[28]
        other2ego_rt = inputs[29]
        fused, added_query, added_reference = self.agent.fuse_agent_tracks_trt(
            inf_tracks, veh_tracks, match_vehicle_index, other2ego_rt
        )
        return tuple(fused) + (added_query, added_reference)


class BevAugmentation(torch.nn.Module):
    def __init__(self, agent):
        super().__init__()
        self.agent = agent

    def forward(self, bev_embed, bev_pos, added_query, added_reference):
        return self.agent.augment_track_bev_trt(
            bev_embed, bev_pos, added_query, added_reference
        )


def build_engine(onnx_path, engine_path, plugin, precision):
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
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * 1024 ** 3)
    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - start
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    with open(engine_path, "wb") as handle:
        handle.write(serialized)
    return build_seconds, network.num_layers


def tensor_statistics(value):
    value = value.detach().float().cpu()
    finite = torch.isfinite(value)
    result = {
        "finite_count": int(finite.sum()),
        "element_count": int(value.numel()),
    }
    if finite.any():
        finite_value = value[finite]
        result.update({
            "finite_min": float(finite_value.min()),
            "finite_max": float(finite_value.max()),
            "finite_mean_abs": float(finite_value.abs().mean()),
        })
    return result


def reference_grid_statistics(value):
    value = value.detach().float().cpu()
    xy = (value[:, :2].sigmoid() * 200).long()
    unique, counts = torch.unique(xy, dim=0, return_counts=True)
    return {
        "reference_count": int(xy.shape[0]),
        "unique_xy_cell_count": int(unique.shape[0]),
        "maximum_xy_cell_multiplicity": int(counts.max()) if counts.numel() else 0,
        "xy_cells": xy.tolist(),
    }


def compare_outputs(names, reference, actual):
    comparison = {}
    for name, expected in zip(names, reference):
        observed = actual[name].cpu()
        expected = expected.detach().cpu()
        item = {
            "reference_shape": list(expected.shape),
            "actual_shape": list(observed.shape),
            "reference_statistics": tensor_statistics(expected),
            "actual_statistics": tensor_statistics(observed),
        }
        if expected.shape == observed.shape:
            if expected.is_floating_point():
                difference = (expected.float() - observed.float()).abs()
                item.update({
                    "mean_abs_error": float(difference.mean()),
                    "max_abs_error": float(difference.max()),
                    "allclose_rtol_1e-3_atol_1e-4": bool(torch.allclose(
                        expected.float(), observed.float(), rtol=1.0e-3, atol=1.0e-4
                    )),
                })
            else:
                item["mismatch_count"] = int((expected != observed).sum())
        comparison[name] = item
    return comparison


def export_build_compare(
    model, inputs, input_names, output_names, stem, plugin, precision
):
    onnx_path = stem + ".onnx"
    engine_path = stem + ".engine"
    with torch.no_grad():
        reference = tuple(model(*inputs))
    torch.onnx.export(
        model,
        inputs,
        onnx_path,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=16,
    )
    graph = onnx.load(onnx_path)
    scatter_reductions = []
    for node in graph.graph.node:
        if node.op_type != "ScatterElements":
            continue
        node.attribute.append(onnx.helper.make_attribute("reduction", "add"))
        scatter_reductions.append(node.name)
    if scatter_reductions:
        onnx.save(graph, onnx_path)
    build_seconds, layers = build_engine(
        onnx_path, engine_path, plugin, precision
    )
    runtime = TensorRTEngine(engine_path, plugin)
    actual = runtime.infer(dict(zip(input_names, inputs)))
    return {
        "precision": precision,
        "build_seconds": build_seconds,
        "network_layers": layers,
        "scatter_add_reductions": scatter_reductions,
        "outputs": compare_outputs(output_names, reference, actual),
    }, reference, actual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("coop_input_npz")
    parser.add_argument("plugin")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    _, agent = build_trt_agent(args.config, args.checkpoint, "ego")
    agent = agent.cuda().eval()
    device = torch.device("cuda")
    coop = np.load(args.coop_input_npz)
    inf_tracks = tuple(
        torch.from_numpy(coop[f"coop_track_intances{index}"]).to(device)
        for index in range(14)
    )
    veh_tracks = tuple(empty_track_state(device))
    match = torch.from_numpy(coop["coop_match_vehicle_index"]).to(device)
    transform = torch.from_numpy(coop["coop_other2ego_rt"]).to(device)

    fusion_inputs = inf_tracks + veh_tracks + (match, transform)
    fusion_input_names = [
        *[f"inf_track_{index}" for index in range(14)],
        *[f"veh_track_{index}" for index in range(14)],
        "match_vehicle_index",
        "other2ego_rt",
    ]
    fusion_output_names = [
        *[f"fused_track_{index}" for index in range(14)],
        "added_query",
        "added_reference",
    ]
    fusion_report, fusion_reference, fusion_actual = export_build_compare(
        AgentFusion(agent),
        fusion_inputs,
        fusion_input_names,
        fusion_output_names,
        os.path.join(args.output_dir, "agent_fusion"),
        args.plugin,
        args.precision,
    )
    fusion_report["reference_grid"] = reference_grid_statistics(
        fusion_reference[-1]
    )
    fusion_report["actual_grid"] = reference_grid_statistics(
        fusion_actual["added_reference"]
    )

    added_query, added_reference = fusion_reference[-2:]
    bev_embed = torch.zeros(40000, 1, 256, device=device)
    bev_pos = torch.zeros(1, 256, 200, 200, device=device)
    augmentation_report, _, _ = export_build_compare(
        BevAugmentation(agent),
        (bev_embed, bev_pos, added_query, added_reference),
        ("bev_embed", "bev_pos", "added_query", "added_reference"),
        ("bev_embed_out", "bev_pos_out"),
        os.path.join(args.output_dir, "bev_augmentation"),
        args.plugin,
        args.precision,
    )

    report = {
        "coop_input_npz": os.path.abspath(args.coop_input_npz),
        "cooperative_track_count": int(inf_tracks[0].shape[0]),
        "precision": args.precision,
        "agent_fusion": fusion_report,
        "bev_augmentation": augmentation_report,
    }
    report_path = os.path.join(args.output_dir, "comparison.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
