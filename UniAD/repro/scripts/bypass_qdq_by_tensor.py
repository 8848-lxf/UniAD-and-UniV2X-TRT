#!/usr/bin/env python3

"""Bypass selected activation Q/DQ pairs while preserving the FP tensor path."""

import argparse
import json
from pathlib import Path

import onnx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tensor", required=True, action="append")
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output ONNX paths must differ")

    model = onnx.load(args.input)
    graph = model.graph
    nodes = list(graph.node)
    graph_outputs = {value.name for value in graph.output}
    removed_ids = set()
    rewrites = []

    for tensor_name in args.tensor:
        quantizers = [
            node for node in nodes
            if node.op_type == "QuantizeLinear" and node.input[0] == tensor_name
        ]
        if len(quantizers) != 1:
            raise ValueError(
                f"Expected one QuantizeLinear for {tensor_name}, got {len(quantizers)}"
            )
        quantizer = quantizers[0]
        quantized_name = quantizer.output[0]
        dequantizers = [
            node for node in nodes
            if node.op_type == "DequantizeLinear"
            and node.input[0] == quantized_name
        ]
        if len(dequantizers) != 1:
            raise ValueError(
                f"Expected one DequantizeLinear for {tensor_name}, got {len(dequantizers)}"
            )
        dequantizer = dequantizers[0]
        dequantized_name = dequantizer.output[0]
        if quantized_name in graph_outputs or dequantized_name in graph_outputs:
            raise ValueError(f"Refusing to bypass a graph-output Q/DQ chain: {tensor_name}")

        consumer_count = 0
        for node in nodes:
            if node is dequantizer:
                continue
            for index, input_name in enumerate(node.input):
                if input_name == dequantized_name:
                    node.input[index] = tensor_name
                    consumer_count += 1
        if consumer_count == 0:
            raise ValueError(f"Dequantized tensor has no consumers: {tensor_name}")

        removed_ids.update((id(quantizer), id(dequantizer)))
        rewrites.append(
            {
                "tensor": tensor_name,
                "quantizer": quantizer.name,
                "dequantizer": dequantizer.name,
                "scale": quantizer.input[1],
                "zero_point": quantizer.input[2] if len(quantizer.input) > 2 else None,
                "rewired_consumer_inputs": consumer_count,
            }
        )

    kept_nodes = [node for node in nodes if id(node) not in removed_ids]
    del graph.node[:]
    graph.node.extend(kept_nodes)

    referenced_initializers = {
        input_name for node in graph.node for input_name in node.input
    }
    kept_initializers = [
        initializer for initializer in graph.initializer
        if initializer.name in referenced_initializers
    ]
    del graph.initializer[:]
    graph.initializer.extend(kept_initializers)

    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, args.output)

    report = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "bypassed_activation_qdq": rewrites,
        "removed_nodes": len(removed_ids),
        "remaining_quantize_linear": sum(
            node.op_type == "QuantizeLinear" for node in graph.node
        ),
        "remaining_dequantize_linear": sum(
            node.op_type == "DequantizeLinear" for node in graph.node
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
