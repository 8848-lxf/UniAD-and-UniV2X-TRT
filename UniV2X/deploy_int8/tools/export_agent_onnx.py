import argparse
import collections
import json
import os

import onnx
import onnx_graphsurgeon as gs
import torch
from torch.onnx import OperatorExportTypes

import projects.mmdet3d_plugin  # noqa: F401

from trt_runtime import (
    INFRASTRUCTURE_OUTPUT_NAMES,
    EGO_INPUT_NAMES,
    INPUT_NAMES,
    OUTPUT_NAMES,
    build_trt_agent,
    load_export_inputs,
)


def dynamic_axes(input_names, output_names):
    axes = {
        name: {0: "track_count"}
        for name in INPUT_NAMES[:14]
    }
    axes.update({
        "gt_lane_labels": {1: "lane_count"},
        "gt_lane_masks": {1: "lane_count"},
    })
    for name in output_names[:14]:
        axes[name] = {0: "next_track_count"}
    if "coop_track_intances0" in input_names:
        for name in EGO_INPUT_NAMES[len(INPUT_NAMES):len(INPUT_NAMES) + 14]:
            axes[name] = {0: "coop_track_count"}
        axes["coop_match_vehicle_index"] = {0: "coop_track_count"}
    for name in (
        "bboxes_dict_bboxes",
        "scores",
        "labels",
        "bbox_index",
        "obj_idxes",
    ):
        axes[name] = {0: "detection_count"}
    for name in ("det_bboxes", "det_scores", "det_labels"):
        axes[name] = {0: "detector_query_count"}
    return axes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("input_npz")
    parser.add_argument("--agent", choices=("ego", "infrastructure"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--coop-input-npz")
    args = parser.parse_args()

    _, model = build_trt_agent(args.config, args.checkpoint, args.agent)
    model = model.cuda().eval()
    model.forward = model.forward_uniad_trt
    inputs = load_export_inputs(
        args.input_npz, torch.device("cuda"), args.coop_input_npz
    )
    output_names = (
        OUTPUT_NAMES if args.agent == "ego" else INFRASTRUCTURE_OUTPUT_NAMES
    )
    input_names = EGO_INPUT_NAMES if args.agent == "ego" else INPUT_NAMES
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with torch.no_grad():
        reference = model(*inputs)
    if len(reference) != len(output_names):
        raise RuntimeError((len(reference), len(output_names)))

    torch.onnx.export(
        model,
        inputs,
        args.output,
        export_params=True,
        keep_initializers_as_inputs=True,
        do_constant_folding=False,
        input_names=input_names,
        output_names=output_names,
        opset_version=16,
        operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
        dynamic_axes=dynamic_axes(input_names, output_names),
        verbose=False,
    )

    graph = gs.import_onnx(onnx.load(args.output, load_external_data=False))
    for node in graph.nodes:
        if node.op == "Reshape":
            node.attrs["allowzero"] = 1
    graph.cleanup().toposort()
    onnx.save_model(
        gs.export_onnx(graph),
        args.output,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(args.output) + ".data",
        size_threshold=1024,
    )
    checked = onnx.load(args.output, load_external_data=True)
    op_counts = collections.Counter(
        (node.domain, node.op_type) for node in checked.graph.node
    )

    report = {
        "agent": args.agent,
        "onnx": os.path.abspath(args.output),
        "onnx_bytes": os.path.getsize(args.output),
        "external_data_bytes": os.path.getsize(args.output + ".data"),
        "inputs": [item.name for item in checked.graph.input],
        "outputs": [item.name for item in checked.graph.output],
        "nodes": len(checked.graph.node),
        "custom_ops": {
            (domain + "::" + op_type if domain else op_type): count
            for (domain, op_type), count in op_counts.items()
            if domain or op_type.endswith("TRT")
        },
        "opsets": {item.domain: item.version for item in checked.opset_import},
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
