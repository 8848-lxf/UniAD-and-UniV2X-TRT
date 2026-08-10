import argparse
import collections
import json
import os

import numpy as np
import onnx


PLUGIN_REWRITES = {
    ("mmcv", "MMCVModulatedDeformConv2d"):
        ("", "ModulatedDeformableConv2dTRT"),
}

MAP_METRIC_OUTPUTS = {
    "drivable_intersection",
    "drivable_union",
    "lanes_intersection",
    "lanes_union",
    "divider_intersection",
    "divider_union",
    "crossing_intersection",
    "crossing_union",
    "contour_intersection",
    "contour_union",
}


def prune_to_outputs(model, dropped_outputs):
    original_outputs = [value.name for value in model.graph.output]
    retained_outputs = [
        value for value in model.graph.output
        if value.name not in dropped_outputs
    ]
    missing = sorted(dropped_outputs - set(original_outputs))
    if missing:
        raise RuntimeError("ONNX outputs not found: %s" % missing)

    required_tensors = {value.name for value in retained_outputs}
    retained_nodes_reversed = []
    for node in reversed(model.graph.node):
        if not any(name in required_tensors for name in node.output):
            continue
        retained_nodes_reversed.append(node)
        required_tensors.update(name for name in node.input if name)

    original_counts = {
        "nodes": len(model.graph.node),
        "inputs": len(model.graph.input),
        "outputs": len(model.graph.output),
        "initializers": len(model.graph.initializer),
    }
    retained_nodes = list(reversed(retained_nodes_reversed))
    retained_inputs = [
        value for value in model.graph.input
        if value.name in required_tensors
    ]
    retained_initializers = [
        value for value in model.graph.initializer
        if value.name in required_tensors
    ]
    retained_value_info = [
        value for value in model.graph.value_info
        if value.name in required_tensors
    ]

    del model.graph.node[:]
    model.graph.node.extend(retained_nodes)
    del model.graph.input[:]
    model.graph.input.extend(retained_inputs)
    del model.graph.output[:]
    model.graph.output.extend(retained_outputs)
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained_initializers)
    del model.graph.value_info[:]
    model.graph.value_info.extend(retained_value_info)

    return {
        "dropped_outputs": sorted(dropped_outputs),
        "removed_inputs": sorted(
            set(value.name for value in model.graph.input)
            ^ set(value.name for value in retained_inputs)
        ),
        "before": original_counts,
        "after": {
            "nodes": len(retained_nodes),
            "inputs": len(retained_inputs),
            "outputs": len(retained_outputs),
            "initializers": len(retained_initializers),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--report", required=True)
    parser.add_argument("--promote-uint8-input", action="append", default=[])
    parser.add_argument("--promote-uint8-casts", action="store_true")
    parser.add_argument("--repair-index-add-scatter", action="store_true")
    parser.add_argument("--materialize-camera-slots", action="store_true")
    parser.add_argument("--drop-map-metric-outputs", action="store_true")
    args = parser.parse_args()

    source = os.path.abspath(args.input)
    output = os.path.abspath(args.output)
    model = onnx.load(source, load_external_data=True)
    rewrites = collections.Counter()
    for node in model.graph.node:
        key = (node.domain, node.op_type)
        if key in PLUGIN_REWRITES:
            node.domain, node.op_type = PLUGIN_REWRITES[key]
            rewrites[key] += 1

    camera_slot_tiles = []
    if args.materialize_camera_slots:
        camera_embeddings = {
            initializer.name for initializer in model.graph.initializer
            if initializer.name.endswith("pts_bbox_head.transformer.cams_embeds")
        }
        if len(camera_embeddings) != 1:
            raise RuntimeError(
                "Expected one camera embedding initializer, got %s"
                % sorted(camera_embeddings)
            )
        camera_embedding = next(iter(camera_embeddings))
        camera_reshape_outputs = {
            node.output[0]
            for node in model.graph.node
            if node.op_type == "Reshape"
            and node.input
            and node.input[0] == camera_embedding
        }
        repeats_name = "univ2x_camera_slot_repeats"
        model.graph.initializer.append(onnx.numpy_helper.from_array(
            np.asarray([6, 1, 1, 1], dtype=np.int64),
            name=repeats_name,
        ))
        rewritten_nodes = []
        for node in model.graph.node:
            camera_inputs = [
                index for index, name in enumerate(node.input)
                if name in camera_reshape_outputs
            ]
            if node.op_type == "Add" and len(camera_inputs) == 1:
                camera_index = camera_inputs[0]
                feature_index = 1 - camera_index
                feature_name = node.input[feature_index]
                tiled_name = feature_name + "_univ2x_six_slots"
                tile_name = (node.name or "camera_add") + "_MaterializeCameraSlots"
                rewritten_nodes.append(onnx.helper.make_node(
                    "Tile",
                    [feature_name, repeats_name],
                    [tiled_name],
                    name=tile_name,
                ))
                node.input[feature_index] = tiled_name
                camera_slot_tiles.append({
                    "add": node.name,
                    "feature_input": feature_name,
                    "tiled_output": tiled_name,
                    "tile": tile_name,
                })
            rewritten_nodes.append(node)
        del model.graph.node[:]
        model.graph.node.extend(rewritten_nodes)

    input_promotions = {}
    graph_inputs = {value.name: value for value in model.graph.input}
    for name in args.promote_uint8_input:
        if name not in graph_inputs:
            raise RuntimeError("ONNX input not found: %s" % name)
        tensor_type = graph_inputs[name].type.tensor_type
        if tensor_type.elem_type != onnx.TensorProto.UINT8:
            raise RuntimeError(
                "Expected UINT8 input %s, got %s" % (
                    name,
                    onnx.TensorProto.DataType.Name(tensor_type.elem_type),
                )
            )
        tensor_type.elem_type = onnx.TensorProto.INT32
        input_promotions[name] = {"from": "UINT8", "to": "INT32"}

    cast_promotions = []
    if args.promote_uint8_casts:
        for node in model.graph.node:
            if node.op_type != "Cast":
                continue
            for attribute in node.attribute:
                if (attribute.name == "to"
                        and attribute.i == onnx.TensorProto.UINT8):
                    attribute.i = onnx.TensorProto.INT32
                    cast_promotions.append(node.name)

    scatter_reductions = []
    if args.repair_index_add_scatter:
        for node in model.graph.node:
            if node.op_type != "ScatterElements":
                continue
            reductions = [
                attribute for attribute in node.attribute
                if attribute.name == "reduction"
            ]
            if reductions:
                if onnx.helper.get_attribute_value(reductions[0]) != b"add":
                    raise RuntimeError(
                        "Unexpected ScatterElements reduction on %s" % node.name
                    )
                continue
            node.attribute.append(onnx.helper.make_attribute("reduction", "add"))
            scatter_reductions.append(node.name)

    graph_pruning = None
    if args.drop_map_metric_outputs:
        original_inputs = {value.name for value in model.graph.input}
        graph_pruning = prune_to_outputs(model, MAP_METRIC_OUTPUTS)
        retained_inputs = {value.name for value in model.graph.input}
        graph_pruning["removed_inputs"] = sorted(original_inputs - retained_inputs)
        expected_removed = {"gt_lane_labels", "gt_lane_masks"}
        if set(graph_pruning["removed_inputs"]) != expected_removed:
            raise RuntimeError(graph_pruning)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    external_name = os.path.basename(output) + ".data"
    for generated_path in (output, output + ".data"):
        if os.path.isfile(generated_path):
            os.remove(generated_path)
    onnx.save_model(
        model,
        output,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_name,
        size_threshold=1024,
    )

    verified = onnx.load(output, load_external_data=True)
    verified_inputs = {value.name: value for value in verified.graph.input}
    for name in input_promotions:
        if args.drop_map_metric_outputs and name not in verified_inputs:
            continue
        if (verified_inputs[name].type.tensor_type.elem_type
                != onnx.TensorProto.INT32):
            raise RuntimeError("Input promotion was not persisted: %s" % name)
    remaining_uint8_casts = [
        node.name
        for node in verified.graph.node
        if node.op_type == "Cast"
        and any(
            attribute.name == "to"
            and attribute.i == onnx.TensorProto.UINT8
            for attribute in node.attribute
        )
    ]
    remaining = collections.Counter(
        (node.domain, node.op_type) for node in verified.graph.node
    )
    unresolved = {
        (domain + "::" + op_type if domain else op_type): count
        for (domain, op_type), count in remaining.items()
        if (domain, op_type) in PLUGIN_REWRITES
    }
    report = {
        "source": source,
        "output": output,
        "rewrites": {
            (domain + "::" + op_type): count
            for (domain, op_type), count in rewrites.items()
        },
        "unresolved": unresolved,
        "input_promotions": input_promotions,
        "uint8_cast_promotions": cast_promotions,
        "remaining_uint8_casts": remaining_uint8_casts,
        "scatter_add_reductions": scatter_reductions,
        "camera_slot_tiles": camera_slot_tiles,
        "graph_pruning": graph_pruning,
        "onnx_bytes": os.path.getsize(output),
        "external_data_bytes": os.path.getsize(output + ".data"),
    }
    if (args.promote_uint8_casts
            and (len(cast_promotions) != 1 or remaining_uint8_casts)):
        raise RuntimeError(report)
    if args.repair_index_add_scatter and not scatter_reductions:
        raise RuntimeError(report)
    if args.materialize_camera_slots and len(camera_slot_tiles) != 4:
        raise RuntimeError(report)
    if unresolved or rewrites != collections.Counter({
        ("mmcv", "MMCVModulatedDeformConv2d"): 26
    }):
        raise RuntimeError(report)
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
