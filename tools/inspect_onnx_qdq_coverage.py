#!/usr/bin/env python3
"""Summarize explicit-QDQ coverage without loading external tensor data."""

import argparse
import collections
import json
import os

import onnx
from onnx import TensorProto


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx")
    parser.add_argument("--output")
    return parser.parse_args()


def sorted_counts(counter):
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def main():
    args = parse_args()
    model = onnx.load(args.onnx, load_external_data=False)
    nodes = list(model.graph.node)
    producers = {
        output: node for node in nodes for output in node.output if output
    }
    consumers = collections.defaultdict(list)
    for node in nodes:
        for value in node.input:
            consumers[value].append(node)

    operator_counts = collections.Counter(node.op_type for node in nodes)
    int8_initializers = {
        value.name for value in model.graph.initializer
        if value.data_type in (TensorProto.INT8, TensorProto.UINT8)
    }
    dq_adjacent = collections.Counter()
    quantized_weight_consumers = collections.Counter()
    q_input_producers = collections.Counter()
    custom_qdq_consumers = collections.Counter()

    for node in nodes:
        if node.op_type == "DequantizeLinear" and node.output:
            downstream = consumers.get(node.output[0], [])
            for consumer in downstream:
                dq_adjacent[consumer.op_type] += 1
                if consumer.domain:
                    custom_qdq_consumers[
                        "%s::%s" % (consumer.domain, consumer.op_type)
                    ] += 1
            if node.input and node.input[0] in int8_initializers:
                for consumer in downstream:
                    quantized_weight_consumers[consumer.op_type] += 1
        elif node.op_type == "QuantizeLinear" and node.input:
            producer = producers.get(node.input[0])
            if producer is not None:
                q_input_producers[producer.op_type] += 1

    weighted_ops = ("Conv", "Gemm", "MatMul")
    weighted_coverage = {}
    for op_type in weighted_ops:
        total = operator_counts.get(op_type, 0)
        quantized = quantized_weight_consumers.get(op_type, 0)
        weighted_coverage[op_type] = {
            "total_nodes": total,
            "int8_weight_nodes": quantized,
            "int8_weight_fraction": float(quantized) / total if total else None,
        }

    report = {
        "schema_version": 1,
        "onnx": os.path.realpath(args.onnx),
        "nodes": len(nodes),
        "quantize_linear_nodes": operator_counts.get("QuantizeLinear", 0),
        "dequantize_linear_nodes": operator_counts.get("DequantizeLinear", 0),
        "int8_initializers": len(int8_initializers),
        "weighted_operator_coverage": weighted_coverage,
        "int8_weight_consumer_types": sorted_counts(quantized_weight_consumers),
        "dequantize_output_consumer_types": sorted_counts(dq_adjacent),
        "quantize_input_producer_types": sorted_counts(q_input_producers),
        "custom_domain_qdq_consumer_types": sorted_counts(custom_qdq_consumers),
        "interpretation": (
            "INT8 weight coverage proves explicit quantized weights for listed "
            "operators. QDQ adjacency alone does not prove that TensorRT selected "
            "a native INT8 kernel for an elementwise, shape, or custom-plugin layer."
        ),
    }
    payload = json.dumps(report, indent=2)
    if args.output:
        directory = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(directory, exist_ok=True)
        with open(args.output, "w") as handle:
            handle.write(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
