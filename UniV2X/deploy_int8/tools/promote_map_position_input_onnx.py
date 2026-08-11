import argparse
import json
import os

import onnx
import onnx_graphsurgeon as gs
from onnx import helper, TensorProto


INPUT_NAME = "map_position_encoding"


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
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    source = os.path.abspath(args.input)
    output = os.path.abspath(args.output)
    model = onnx.load(source, load_external_data=False)
    if any(value.name == INPUT_NAME for value in model.graph.input):
        raise RuntimeError("Graph already has %s" % INPUT_NAME)

    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    level_embeddings = [
        initializer.name
        for initializer in model.graph.initializer
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
    for index, name in enumerate(add.input):
        if name == old_position:
            add.input[index] = INPUT_NAME
    model.graph.input.append(helper.make_tensor_value_info(
        INPUT_NAME, TensorProto.FLOAT, [1, 40000, 256]
    ))

    node_count_before_cleanup = len(model.graph.node)
    graph = gs.import_onnx(model)
    graph.cleanup().toposort()
    model = gs.export_onnx(graph)
    node_count_after_cleanup = len(model.graph.node)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    onnx.save(model, output)
    verified = onnx.load(output, load_external_data=False)
    inputs = {
        value.name: [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        for value in verified.graph.input
    }
    patched_add = [node for node in verified.graph.node if node.name == add.name]
    if inputs.get(INPUT_NAME) != [1, 40000, 256]:
        raise RuntimeError("Map position input shape was not persisted")
    if len(patched_add) != 1 or INPUT_NAME not in patched_add[0].input:
        raise RuntimeError("Map position input was not wired to the target Add")

    report = {
        "source": source,
        "output": output,
        "input_name": INPUT_NAME,
        "input_shape": inputs[INPUT_NAME],
        "input_dtype": "FLOAT",
        "level_embedding_initializer": level_embeddings[0],
        "target_add": add.name,
        "replaced_position_input": old_position,
        "node_count_before_cleanup": node_count_before_cleanup,
        "node_count_after_cleanup": node_count_after_cleanup,
        "dead_nodes_removed": node_count_before_cleanup - node_count_after_cleanup,
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
