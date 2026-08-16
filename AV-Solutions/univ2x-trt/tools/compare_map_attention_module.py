import argparse
import json
import os
import sys

import torch
import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEPLOY_ROOT = os.path.join(REPO_ROOT, "deploy_int8", "UniV2X_deploy")
TRT_FUNCTIONS = os.path.join(
    DEPLOY_ROOT, "projects", "mmdet3d_plugin", "uniad", "functions"
)
for source_root in (REPO_ROOT, DEPLOY_ROOT, TRT_FUNCTIONS):
    while source_root in sys.path:
        sys.path.remove(source_root)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, TRT_FUNCTIONS)
sys.path.insert(0, DEPLOY_ROOT)

import projects.mmdet3d_plugin  # noqa: E402,F401

from compare_msda_plugin import build_engine  # noqa: E402
from trt_engine import TensorRTEngine  # noqa: E402
from trt_runtime import build_trt_agent  # noqa: E402


class MapCrossAttention(torch.nn.Module):
    def __init__(self, attention):
        super().__init__()
        self.attention = attention

    def forward(
        self,
        query,
        value,
        identity,
        query_pos,
        reference_points,
        spatial_shapes,
        level_start_index,
    ):
        return self.attention.forward_trt(
            query,
            value=value,
            identity=identity,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )


class MapDecoderLayer(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(
        self,
        query,
        value,
        identity,
        query_pos,
        reference_points,
        spatial_shapes,
        level_start_index,
    ):
        del identity
        return self.layer(
            query,
            key=value,
            value=value,
            query_pos=query_pos,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )


class MapDecoder(torch.nn.Module):
    def __init__(self, decoder, reg_branches):
        super().__init__()
        self.decoder = decoder
        self.reg_branches = reg_branches

    def forward(
        self,
        query,
        value,
        identity,
        query_pos,
        reference_points,
        spatial_shapes,
        level_start_index,
    ):
        del identity
        valid_ratios = reference_points.new_ones(
            (reference_points.shape[0], spatial_shapes.shape[0], 2)
        )
        return self.decoder(
            query=query,
            key=None,
            value=value,
            query_pos=query_pos,
            reference_points=reference_points[:, :, 0],
            valid_ratios=valid_ratios,
            reg_branches=self.reg_branches,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )


class MapHead(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.head = head

    def forward(self, bev_embed):
        return self.head.forward_trt(bev_embed)


class MapPositionEncoding(torch.nn.Module):
    def __init__(self, positional_encoding):
        super().__init__()
        self.positional_encoding = positional_encoding

    def forward(self, mask):
        return self.positional_encoding(mask)


def error_summary(reference, actual):
    reference = reference.detach().cpu()
    actual = actual.detach().cpu()
    difference = (reference.float() - actual.float()).abs()
    return {
        "mean_abs_error": float(difference.mean()),
        "p99_abs_error": float(torch.quantile(difference.flatten(), 0.99)),
        "max_abs_error": float(difference.max()),
        "allclose_rtol_1e-3_atol_1e-4": bool(torch.allclose(
            reference.float(), actual.float(), rtol=1.0e-3, atol=1.0e-4
        )),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--agent", choices=("ego", "infrastructure"), default="ego")
    parser.add_argument("--query-count", type=int, default=300)
    parser.add_argument(
        "--scope",
        choices=("attention", "layer", "decoder", "head", "posenc"),
        default="attention",
    )
    parser.add_argument("--bev-npz")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(0)
    _, source_model = build_trt_agent(args.config, args.checkpoint, args.agent)
    decoder_layer = source_model.seg_head.transformer.decoder.layers[args.layer]
    attention = decoder_layer.attentions[1]
    if args.scope == "attention":
        model = MapCrossAttention(attention)
    elif args.scope == "layer":
        model = MapDecoderLayer(decoder_layer)
    elif args.scope == "decoder":
        model = MapDecoder(
            source_model.seg_head.transformer.decoder,
            source_model.seg_head.reg_branches,
        )
    elif args.scope == "head":
        model = MapHead(source_model.seg_head)
    else:
        model = MapPositionEncoding(source_model.seg_head.positional_encoding)
    model = model.cuda().eval()

    device = torch.device("cuda")
    query_count = args.query_count
    if args.scope == "posenc":
        inputs = (torch.zeros(1, 200, 200, dtype=torch.bool, device=device),)
        input_names = ("mask",)
    elif args.scope == "head":
        if args.bev_npz:
            with np.load(args.bev_npz) as archive:
                bev = torch.from_numpy(archive["output::bev_embed"]).to(device)
        else:
            bev = torch.randn(40000, 1, 256, device=device)
        inputs = (bev,)
        input_names = ("bev_embed",)
    else:
        inputs = (
            torch.randn(query_count, 1, 256, device=device),
            torch.randn(40000, 1, 256, device=device),
            torch.randn(query_count, 1, 256, device=device),
            torch.randn(query_count, 1, 256, device=device),
            torch.cat([
                torch.rand(1, query_count, 1, 2, device=device) * 0.8 + 0.1,
                torch.rand(1, query_count, 1, 2, device=device) * 0.25 + 0.05,
            ], dim=-1),
            torch.tensor([[200, 200]], dtype=torch.int64, device=device),
            torch.tensor([0], dtype=torch.int64, device=device),
        )
        input_names = (
            "query",
            "value",
            "identity",
            "query_pos",
            "reference_points",
            "spatial_shapes",
            "level_start_index",
        )
    with torch.no_grad():
        reference_values = model(*inputs)
    if not isinstance(reference_values, tuple):
        reference_values = (reference_values,)
    reference_values = tuple(value.detach() for value in reference_values)
    output_names = ["output_%d" % index for index in range(len(reference_values))]

    onnx_path = os.path.join(args.output_dir, "map_cross_attention.onnx")
    engine_path = os.path.join(args.output_dir, "map_cross_attention.engine")
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
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
    )
    build_seconds, network_layers = build_engine(
        onnx_path, engine_path, args.plugin, 2
    )
    runtime = TensorRTEngine(engine_path, args.plugin)
    actual = runtime.infer(dict(zip(input_names, inputs)))
    report = {
        "agent": args.agent,
        "scope": args.scope,
        "decoder_layer": args.layer,
        "configured_levels": int(attention.num_levels),
        "runtime_levels": 1,
        "query_count": query_count,
        "bev_npz": os.path.abspath(args.bev_npz) if args.bev_npz else None,
        "build_seconds": build_seconds,
        "network_layers": network_layers,
        "numerics": {
            name: error_summary(reference, actual[name])
            for name, reference in zip(output_names, reference_values)
        },
        "onnx": os.path.abspath(onnx_path),
        "engine": os.path.abspath(engine_path),
    }
    report_path = os.path.join(args.output_dir, "comparison.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
