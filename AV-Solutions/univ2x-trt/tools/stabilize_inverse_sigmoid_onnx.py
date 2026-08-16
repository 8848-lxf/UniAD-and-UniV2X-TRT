import argparse
import hashlib
import json
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--node-name",
        action="append",
        default=[],
        help="Rewrite only the named Log node; may be repeated. Defaults to all matches.",
    )
    parser.add_argument(
        "--constant-value",
        action="append",
        default=[],
        metavar="NODE=VALUE",
        help="Replace a scalar Constant tensor while preserving its dtype.",
    )
    parser.add_argument(
        "--constants-only",
        action="store_true",
        help="Skip inverse-sigmoid rewrites and only apply --constant-value edits.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.realpath(args.input)
    output_path = os.path.realpath(args.output)
    if input_path == output_path:
        raise ValueError("Input and output ONNX paths must differ")

    model = onnx.load(input_path, load_external_data=True)
    requested_nodes = set(args.node_name)
    requested_constants = {}
    for spec in args.constant_value:
        name, separator, raw_value = spec.partition("=")
        if not separator or not name:
            raise ValueError("Expected --constant-value NODE=VALUE, got %r" % spec)
        requested_constants[name] = float(raw_value)
    replaced_constants = []
    for node in model.graph.node:
        if node.name not in requested_constants:
            continue
        if node.op_type != "Constant":
            raise RuntimeError("Requested node is not Constant: %s" % node.name)
        value_attributes = [attribute for attribute in node.attribute if attribute.name == "value"]
        if len(value_attributes) != 1:
            raise RuntimeError("Constant node has no unique tensor value: %s" % node.name)
        attribute = value_attributes[0]
        old_array = numpy_helper.to_array(attribute.t)
        if old_array.size != 1:
            raise RuntimeError("Constant node is not scalar: %s" % node.name)
        new_array = np.asarray(requested_constants[node.name], dtype=old_array.dtype)
        attribute.t.CopyFrom(numpy_helper.from_array(new_array))
        replaced_constants.append({
            "node": node.name,
            "dtype": str(old_array.dtype),
            "old_value": float(old_array.reshape(-1)[0]),
            "new_value": float(new_array.reshape(-1)[0]),
        })
    missing_constants = set(requested_constants) - {
        item["node"] for item in replaced_constants
    }
    if missing_constants:
        raise RuntimeError("Requested Constant nodes were not found: %s" % sorted(missing_constants))
    producers = {
        output: node
        for node in model.graph.node
        for output in node.output
    }
    replacements = []
    rewritten_nodes = []
    for index, node in enumerate(model.graph.node):
        if args.constants_only:
            replacements.append(node)
            continue
        if node.op_type != "Log" or len(node.input) != 1:
            replacements.append(node)
            continue
        div = producers.get(node.input[0])
        if div is None or div.op_type != "Div" or len(div.input) != 2:
            replacements.append(node)
            continue
        if requested_nodes and node.name not in requested_nodes:
            replacements.append(node)
            continue

        numerator_log = "%s__stable_numerator_log" % node.output[0]
        denominator_log = "%s__stable_denominator_log" % node.output[0]
        replacements.extend([
            helper.make_node(
                "Log", [div.input[0]], [numerator_log],
                name="%s_StableNumeratorLog" % node.name,
            ),
            helper.make_node(
                "Log", [div.input[1]], [denominator_log],
                name="%s_StableDenominatorLog" % node.name,
            ),
            helper.make_node(
                "Sub", [numerator_log, denominator_log], list(node.output),
                name="%s_StableSubtract" % node.name,
            ),
        ])
        rewritten_nodes.append({
            "original_log": node.name,
            "original_div": div.name,
            "output": node.output[0],
            "numerator": div.input[0],
            "denominator": div.input[1],
        })

    if not rewritten_nodes and not replaced_constants:
        raise RuntimeError(
            "No Log(Div(a,b)) patterns or requested scalar constants were rewritten"
        )
    rewritten_names = {item["original_log"] for item in rewritten_nodes}
    missing_nodes = requested_nodes - rewritten_names
    if missing_nodes:
        raise RuntimeError("Requested Log nodes were not rewritten: %s" % sorted(missing_nodes))
    del model.graph.node[:]
    model.graph.node.extend(replacements)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    external_name = os.path.basename(output_path) + ".data"
    onnx.save_model(
        model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_name,
        size_threshold=1024,
    )
    saved = onnx.load(output_path, load_external_data=False)
    if len(saved.graph.node) != len(model.graph.node):
        raise RuntimeError("Saved ONNX node count changed unexpectedly")

    report = {
        "input": input_path,
        "input_sha256": sha256(input_path),
        "output": output_path,
        "output_sha256": sha256(output_path),
        "external_data": os.path.join(os.path.dirname(output_path), external_name),
        "rewrite": "log(a / b) -> log(a) - log(b)",
        "requested_nodes": sorted(requested_nodes),
        "pattern_count": len(rewritten_nodes),
        "patterns": rewritten_nodes,
        "replaced_constants": replaced_constants,
    }
    report_path = os.path.realpath(args.report)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
