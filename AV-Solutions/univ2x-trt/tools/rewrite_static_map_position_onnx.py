import argparse
import json
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper


POSITION_SHAPE = [1, 256, 200, 200]


def constant_tensor(node):
    if node.op_type != "Constant":
        return None
    for attribute in node.attribute:
        if attribute.name == "value":
            return attribute.t
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    source = os.path.abspath(args.input)
    output = os.path.abspath(args.output)
    model = onnx.load(source, load_external_data=False)
    matches = []
    for index, node in enumerate(model.graph.node):
        tensor = constant_tensor(node)
        if tensor is not None and list(tensor.dims) == POSITION_SHAPE:
            matches.append((index, node, tensor))
    if len(matches) != 1:
        raise RuntimeError(
            "Expected one static map position constant, found %d" % len(matches)
        )

    index, node, tensor = matches[0]
    position = numpy_helper.to_array(tensor).copy()
    y = position[:, :128, :, :1]
    x = position[:, 128:, :1, :]
    reconstructed = np.concatenate(
        [
            np.broadcast_to(y, (1, 128, 200, 200)),
            np.broadcast_to(x, (1, 128, 200, 200)),
        ],
        axis=1,
    )
    if not np.array_equal(position, reconstructed):
        difference = np.abs(position.astype(np.float64) - reconstructed)
        raise RuntimeError({
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
        })

    prefix = node.name or "StaticMapPosition"
    output_name = node.output[0]
    y_name = output_name + "_y"
    x_name = output_name + "_x"
    y_expanded = output_name + "_y_expanded"
    x_expanded = output_name + "_x_expanded"
    shape_name = output_name + "_shape"
    replacements = [
        helper.make_node(
            "Constant",
            [],
            [y_name],
            name=prefix + "_Y",
            value=numpy_helper.from_array(y),
        ),
        helper.make_node(
            "Constant",
            [],
            [x_name],
            name=prefix + "_X",
            value=numpy_helper.from_array(x),
        ),
        helper.make_node(
            "Constant",
            [],
            [shape_name],
            name=prefix + "_Shape",
            value=numpy_helper.from_array(
                np.asarray([1, 128, 200, 200], dtype=np.int64)
            ),
        ),
        helper.make_node(
            "Expand", [y_name, shape_name], [y_expanded], name=prefix + "_ExpandY"
        ),
        helper.make_node(
            "Expand", [x_name, shape_name], [x_expanded], name=prefix + "_ExpandX"
        ),
        helper.make_node(
            "Concat",
            [y_expanded, x_expanded],
            [output_name],
            name=prefix + "_ConcatXY",
            axis=1,
        ),
    ]
    nodes = list(model.graph.node)
    nodes[index:index + 1] = replacements
    del model.graph.node[:]
    model.graph.node.extend(nodes)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    onnx.save(model, output)
    verified = onnx.load(output, load_external_data=False)
    remaining = [
        candidate.name
        for candidate in verified.graph.node
        if constant_tensor(candidate) is not None
        and list(constant_tensor(candidate).dims) == POSITION_SHAPE
    ]
    if remaining:
        raise RuntimeError("Large position constants remain: %s" % remaining)

    report = {
        "source": source,
        "output": output,
        "replaced_node": node.name,
        "replaced_output": output_name,
        "original_constant_bytes": int(position.nbytes),
        "separable_constant_bytes": int(y.nbytes + x.nbytes),
        "exact_reconstruction": True,
        "output_bytes": os.path.getsize(output),
        "external_data_files": sorted({
            entry.value
            for initializer in verified.graph.initializer
            for entry in initializer.external_data
            if entry.key == "location"
        }),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
