import argparse
import json
import os

import numpy as np
import onnx
from onnx import numpy_helper


def scalar_constant(node):
    if node.op_type != "Constant":
        return None
    for attribute in node.attribute:
        if attribute.name == "value":
            value = numpy_helper.to_array(attribute.t)
            if value.size == 1 and np.issubdtype(value.dtype, np.floating):
                return attribute, value
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--source-epsilon", type=float, default=0.0009765625)
    parser.add_argument("--target-epsilon", type=float, default=1e-5)
    parser.add_argument("--expected-pairs", type=int, default=2)
    parser.add_argument("--report")
    args = parser.parse_args()

    if not 0.0 < args.target_epsilon < 0.5:
        raise ValueError("--target-epsilon must be between zero and 0.5")

    model = onnx.load(args.input, load_external_data=False)
    replacements = []
    targets = {
        args.source_epsilon: args.target_epsilon,
        1.0 - args.source_epsilon: 1.0 - args.target_epsilon,
    }
    for node in model.graph.node:
        result = scalar_constant(node)
        if result is None:
            continue
        attribute, value = result
        scalar = float(value.reshape(-1)[0])
        for source, target in targets.items():
            if not np.isclose(scalar, source, rtol=0.0, atol=1e-9):
                continue
            replacement = np.asarray(target, dtype=value.dtype).reshape(value.shape)
            attribute.t.CopyFrom(numpy_helper.from_array(replacement))
            replacements.append({
                "node": node.name,
                "output": list(node.output),
                "source": scalar,
                "target": float(replacement.reshape(-1)[0]),
            })
            break

    expected = args.expected_pairs * 2
    if len(replacements) != expected:
        raise RuntimeError(
            "Expected %d normalization constants, found %d: %s"
            % (expected, len(replacements), replacements)
        )

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    onnx.save(model, output)
    verified = onnx.load(output, load_external_data=True)
    replacement_nodes = {item["node"]: item["target"] for item in replacements}
    verified_targets = []
    for node in verified.graph.node:
        if node.name not in replacement_nodes:
            continue
        result = scalar_constant(node)
        if result is None:
            continue
        _, value = result
        scalar = float(value.reshape(-1)[0])
        if np.isclose(
            scalar, replacement_nodes[node.name], rtol=0.0, atol=1e-9
        ):
            verified_targets.append(node.name)
    if len(verified_targets) != expected:
        raise RuntimeError(
            "Patched graph only retained %d of %d target constants"
            % (len(verified_targets), expected)
        )

    external_data_files = sorted({
        entry.value
        for initializer in model.graph.initializer
        for entry in initializer.external_data
        if entry.key == "location"
    })
    result = {
        "source": os.path.abspath(args.input),
        "output": output,
        "source_epsilon": args.source_epsilon,
        "target_epsilon": args.target_epsilon,
        "replacements": replacements,
        "verified_target_constants": verified_targets,
        "external_data_files": external_data_files,
    }
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
