#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bypass isolated Q/DQ pairs whose calibration scale is non-finite")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model = onnx.load(args.input)
    graph = model.graph
    initializers = {item.name: item for item in graph.initializer}

    qdq_scale_names = {
        node.input[1]
        for node in graph.node
        if node.op_type in {"QuantizeLinear", "DequantizeLinear"}
        and len(node.input) > 1
    }
    bad_scales = sorted(
        name for name in qdq_scale_names
        if name in initializers
        and not np.isfinite(numpy_helper.to_array(initializers[name])).all()
    )
    if not bad_scales:
        raise RuntimeError("No non-finite Q/DQ scales found")

    nodes_to_remove = set()
    initializer_candidates = set(bad_scales)
    bypasses = []
    for scale_name in bad_scales:
        q_nodes = [
            node for node in graph.node
            if node.op_type == "QuantizeLinear"
            and len(node.input) > 1
            and node.input[1] == scale_name
        ]
        dq_nodes = [
            node for node in graph.node
            if node.op_type == "DequantizeLinear"
            and len(node.input) > 1
            and node.input[1] == scale_name
        ]
        if len(q_nodes) != 1 or len(dq_nodes) != 1:
            raise RuntimeError(
                f"Expected one Q and one DQ for {scale_name}, got "
                f"{len(q_nodes)} and {len(dq_nodes)}")

        q_node, dq_node = q_nodes[0], dq_nodes[0]
        if dq_node.input[0] != q_node.output[0]:
            raise RuntimeError(f"Q/DQ chain is not direct for {scale_name}")
        q_consumers = [
            node for node in graph.node if q_node.output[0] in node.input
        ]
        if [node.name for node in q_consumers] != [dq_node.name]:
            raise RuntimeError(f"Quantized tensor is shared for {scale_name}")
        if dq_node.output[0] in {item.name for item in graph.output}:
            raise RuntimeError(f"DQ output is a graph output for {scale_name}")

        raw_tensor = q_node.input[0]
        dq_tensor = dq_node.output[0]
        consumers = []
        for node in graph.node:
            for index, input_name in enumerate(node.input):
                if input_name == dq_tensor:
                    node.input[index] = raw_tensor
                    consumers.append(node.name)
        if not consumers:
            raise RuntimeError(f"DQ output has no consumers for {scale_name}")

        nodes_to_remove.update({q_node.name, dq_node.name})
        initializer_candidates.update(q_node.input[1:])
        initializer_candidates.update(dq_node.input[1:])
        bypasses.append({
            "scale": scale_name,
            "quantize_node": q_node.name,
            "dequantize_node": dq_node.name,
            "raw_tensor": raw_tensor,
            "consumers_reconnected": consumers,
        })

    kept_nodes = [node for node in graph.node if node.name not in nodes_to_remove]
    del graph.node[:]
    graph.node.extend(kept_nodes)

    still_used = {input_name for node in graph.node for input_name in node.input}
    removed_initializers = sorted(
        name for name in initializer_candidates if name not in still_used)
    kept_initializers = [
        item for item in graph.initializer if item.name not in removed_initializers
    ]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)

    remaining_bad_scales = []
    remaining_initializers = {item.name: item for item in graph.initializer}
    for node in graph.node:
        if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
            continue
        scale_name = node.input[1]
        if scale_name in remaining_initializers:
            scale = numpy_helper.to_array(remaining_initializers[scale_name])
            if not np.isfinite(scale).all():
                remaining_bad_scales.append(scale_name)
    if remaining_bad_scales:
        raise RuntimeError(
            f"Non-finite Q/DQ scales remain: {sorted(set(remaining_bad_scales))}")

    onnx.checker.check_model(model)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)

    report = {
        "schema_version": 1,
        "input": str(Path(args.input).resolve()),
        "output": str(output_path.resolve()),
        "bypassed_qdq_pairs": bypasses,
        "removed_initializers": removed_initializers,
        "remaining_nonfinite_qdq_scales": [],
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as output_file:
        json.dump(report, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


if __name__ == "__main__":
    main()
