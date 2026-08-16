import argparse
import json
import os

import numpy as np
import onnx
from onnx import helper, numpy_helper


POSITION_SHAPE = [1, 256, 200, 200]


def tensor_from_constant(node):
    if node.op_type != "Constant":
        return None
    for attribute in node.attribute:
        if attribute.name == "value":
            return attribute.t
    return None


def only_consumer(consumers, tensor_name, expected_op):
    nodes = consumers.get(tensor_name, [])
    if len(nodes) != 1 or nodes[0].op_type != expected_op:
        raise RuntimeError({
            "tensor": tensor_name,
            "expected_op": expected_op,
            "consumers": [(node.op_type, node.name) for node in nodes],
        })
    return nodes[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runtime_input")
    parser.add_argument("position_source")
    parser.add_argument("output")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    runtime_path = os.path.abspath(args.runtime_input)
    source_path = os.path.abspath(args.position_source)
    output_path = os.path.abspath(args.output)
    runtime = onnx.load(runtime_path, load_external_data=False)
    source = onnx.load(source_path, load_external_data=False)

    position_tensors = []
    for node in source.graph.node:
        tensor = tensor_from_constant(node)
        if tensor is not None and list(tensor.dims) == POSITION_SHAPE:
            position_tensors.append(tensor)
    if len(position_tensors) != 1:
        raise RuntimeError(
            "Expected one exact position source, found %d" % len(position_tensors)
        )
    position = numpy_helper.to_array(position_tensors[0]).copy()
    y = position[:, :128, :, :1].transpose(0, 2, 3, 1).copy()
    x = position[:, 128:, :1, :].transpose(0, 2, 3, 1).copy()
    reconstructed = np.concatenate(
        [
            np.broadcast_to(y, (1, 200, 200, 128)),
            np.broadcast_to(x, (1, 200, 200, 128)),
        ],
        axis=3,
    ).transpose(0, 3, 1, 2)
    if not np.array_equal(position, reconstructed):
        raise RuntimeError("Position source is not exactly separable")

    consumers = {}
    for node in runtime.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    level_embeddings = [
        initializer.name
        for initializer in runtime.graph.initializer
        if initializer.name.endswith("seg_head.transformer.level_embeds")
    ]
    if len(level_embeddings) != 1:
        raise RuntimeError("Unexpected level embedding initializers: %s" % level_embeddings)
    gather = only_consumer(consumers, level_embeddings[0], "Gather")
    reshape = only_consumer(consumers, gather.output[0], "Reshape")
    add = only_consumer(consumers, reshape.output[0], "Add")
    old_position_inputs = [name for name in add.input if name != reshape.output[0]]
    if len(old_position_inputs) != 1:
        raise RuntimeError("Cannot resolve old position input on %s" % add.name)
    old_position = old_position_inputs[0]

    prefix = "ExactStaticMapPosition"
    y_name = prefix + "_Y"
    x_name = prefix + "_X"
    expand_shape_name = prefix + "_ExpandShape"
    reshape_shape_name = prefix + "_ReshapeShape"
    y_expanded = prefix + "_YExpanded"
    x_expanded = prefix + "_XExpanded"
    concatenated = prefix + "_NHWC"
    final_position = prefix + "_LNC"
    replacements = [
        helper.make_node(
            "Constant", [], [y_name], name=prefix + "ConstantY",
            value=numpy_helper.from_array(y),
        ),
        helper.make_node(
            "Constant", [], [x_name], name=prefix + "ConstantX",
            value=numpy_helper.from_array(x),
        ),
        helper.make_node(
            "Constant", [], [expand_shape_name], name=prefix + "ExpandShape",
            value=numpy_helper.from_array(
                np.asarray([1, 200, 200, 128], dtype=np.int64)
            ),
        ),
        helper.make_node(
            "Constant", [], [reshape_shape_name], name=prefix + "ReshapeShape",
            value=numpy_helper.from_array(
                np.asarray([40000, 1, 256], dtype=np.int64)
            ),
        ),
        helper.make_node(
            "Expand", [y_name, expand_shape_name], [y_expanded],
            name=prefix + "ExpandY",
        ),
        helper.make_node(
            "Expand", [x_name, expand_shape_name], [x_expanded],
            name=prefix + "ExpandX",
        ),
        helper.make_node(
            "Concat", [y_expanded, x_expanded], [concatenated],
            name=prefix + "ConcatXY", axis=3,
        ),
        helper.make_node(
            "Reshape", [concatenated, reshape_shape_name], [final_position],
            name=prefix + "ReshapeLNC",
        ),
    ]
    for input_index, name in enumerate(add.input):
        if name == old_position:
            add.input[input_index] = final_position
    add_index = list(runtime.graph.node).index(add)
    nodes = list(runtime.graph.node)
    nodes[add_index:add_index] = replacements
    del runtime.graph.node[:]
    runtime.graph.node.extend(nodes)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save(runtime, output_path)
    verified = onnx.load(output_path, load_external_data=False)
    patched_add = [node for node in verified.graph.node if node.name == add.name]
    if len(patched_add) != 1 or final_position not in patched_add[0].input:
        raise RuntimeError("Patched position input was not persisted")

    report = {
        "runtime_input": runtime_path,
        "position_source": source_path,
        "output": output_path,
        "level_embedding_initializer": level_embeddings[0],
        "target_add": add.name,
        "replaced_position_input": old_position,
        "new_position_input": final_position,
        "exact_reconstruction": True,
        "separable_constant_bytes": int(y.nbytes + x.nbytes),
        "output_bytes": os.path.getsize(output_path),
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
