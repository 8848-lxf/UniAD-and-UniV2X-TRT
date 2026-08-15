import argparse
import ctypes
import hashlib
import json
import os
import re
import time

import tensorrt as trt


class CaptureLogger(trt.ILogger):
    def __init__(self, severity=trt.ILogger.Severity.WARNING):
        super().__init__()
        self.severity = severity
        self.messages = []

    def log(self, severity, message):
        self.messages.append({
            "severity": str(severity).rsplit(".", 1)[-1],
            "message": message,
        })
        if severity <= self.severity:
            print("[TensorRT %s] %s" % (severity, message), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dynamic_bounds(name, shape, args):
    if name.startswith("prev_track_intances"):
        values = (args.track_min, args.track_opt, args.track_max)
    elif name.startswith("coop_track_intances") or name == "coop_match_vehicle_index":
        values = (args.coop_min, args.coop_opt, args.coop_max)
    else:
        raise RuntimeError("No optimization-profile policy for %s: %s" % (name, shape))

    result = []
    for value in values:
        dims = list(shape)
        dynamic = [index for index, dim in enumerate(dims) if dim < 0]
        if dynamic != [0]:
            raise RuntimeError("Unsupported dynamic axes for %s: %s" % (name, shape))
        dims[0] = value
        result.append(tuple(dims))
    return tuple(result)


def tensor_metadata(tensor):
    return {
        "name": tensor.name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def constrain_inverse_sigmoid_to_fp32(network):
    producers = {}
    layers = []
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        layers.append(layer)
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            if tensor is not None:
                producers[tensor.name] = layer

    log_layers = [
        layer for layer in layers
        if layer.type == trt.LayerType.UNARY
        and layer.name.startswith("Log_")
    ]
    traversable = {
        trt.LayerType.UNARY,
        trt.LayerType.ELEMENTWISE,
        trt.LayerType.ACTIVATION,
        trt.LayerType.SHUFFLE,
        trt.LayerType.CONSTANT,
        trt.LayerType.CAST,
        trt.LayerType.DEQUANTIZE,
        trt.LayerType.QUANTIZE,
    }
    constrained_types = {
        trt.LayerType.UNARY,
        trt.LayerType.ELEMENTWISE,
        trt.LayerType.ACTIVATION,
    }
    constrained = {}
    visited = set()
    stack = [(layer, 0) for layer in log_layers]
    while stack:
        layer, depth = stack.pop()
        identity = id(layer)
        if identity in visited or depth > 8:
            continue
        visited.add(identity)
        if layer.type not in traversable:
            continue
        if layer.type in constrained_types:
            layer.precision = trt.float32
            outputs = []
            for output_index in range(layer.num_outputs):
                tensor = layer.get_output(output_index)
                if tensor is not None and tensor.dtype == trt.float32:
                    layer.set_output_type(output_index, trt.float32)
                    outputs.append(tensor.name)
            constrained[layer.name] = {
                "type": str(layer.type),
                "outputs": outputs,
                "distance_from_log": depth,
            }
        for input_index in range(layer.num_inputs):
            tensor = layer.get_input(input_index)
            if tensor is None:
                continue
            producer = producers.get(tensor.name)
            if producer is not None:
                stack.append((producer, depth + 1))
    return {
        "log_layer_count": len(log_layers),
        "constrained_layers": constrained,
    }


def constrain_layernorm_to_fp32(network):
    producers = {}
    consumers = {}
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            if tensor is not None:
                producers[tensor.name] = layer
        for input_index in range(layer.num_inputs):
            tensor = layer.get_input(input_index)
            if tensor is not None:
                consumers.setdefault(tensor.name, []).append(layer)

    def output_names(layer):
        return [
            layer.get_output(index).name
            for index in range(layer.num_outputs)
            if layer.get_output(index) is not None
        ]

    def matching_consumers(tensor_name, prefix):
        return [
            layer for layer in consumers.get(tensor_name, [])
            if layer.name.startswith(prefix)
        ]

    chains = []
    for index in range(network.num_layers):
        power = network.get_layer(index)
        if not power.name.startswith("Pow_") or power.num_inputs < 1:
            continue
        centered_tensor = power.get_input(0)
        centered = producers.get(centered_tensor.name) if centered_tensor else None
        if centered is None or not centered.name.startswith("Sub_"):
            continue
        centered_outputs = output_names(centered)
        power_outputs = output_names(power)
        if len(centered_outputs) != 1 or len(power_outputs) != 1:
            continue

        input_mean = [
            producers.get(centered.get_input(input_index).name)
            for input_index in range(centered.num_inputs)
            if centered.get_input(input_index) is not None
        ]
        input_mean = [
            layer for layer in input_mean
            if layer is not None and layer.name.startswith("ReduceMean_")
        ]
        variance_mean = matching_consumers(power_outputs[0], "ReduceMean_")
        if len(input_mean) != 1 or len(variance_mean) != 1:
            continue
        variance_outputs = output_names(variance_mean[0])
        variance_add = (
            matching_consumers(variance_outputs[0], "Add_")
            if len(variance_outputs) == 1 else []
        )
        if len(variance_add) != 1:
            continue
        variance_add_outputs = output_names(variance_add[0])
        square_root = (
            matching_consumers(variance_add_outputs[0], "Sqrt_")
            if len(variance_add_outputs) == 1 else []
        )
        if len(square_root) != 1:
            continue
        square_root_outputs = output_names(square_root[0])
        division = (
            matching_consumers(square_root_outputs[0], "Div_")
            if len(square_root_outputs) == 1 else []
        )
        division = [
            layer for layer in division
            if any(
                layer.get_input(input_index) is not None
                and layer.get_input(input_index).name == centered_outputs[0]
                for input_index in range(layer.num_inputs)
            )
        ]
        if len(division) != 1:
            continue
        division_outputs = output_names(division[0])
        scale = (
            matching_consumers(division_outputs[0], "Mul_")
            if len(division_outputs) == 1 else []
        )
        if len(scale) != 1:
            continue
        scale_outputs = output_names(scale[0])
        bias = (
            matching_consumers(scale_outputs[0], "Add_")
            if len(scale_outputs) == 1 else []
        )
        if len(bias) != 1:
            continue
        chains.append([
            input_mean[0], centered, power, variance_mean[0], variance_add[0],
            square_root[0], division[0], scale[0], bias[0],
        ])

    constrained = {}
    for chain_index, chain in enumerate(chains):
        chain_names = [layer.name for layer in chain]
        for layer in chain:
            layer.precision = trt.float32
            outputs = []
            for output_index in range(layer.num_outputs):
                tensor = layer.get_output(output_index)
                if tensor is not None and tensor.dtype == trt.float32:
                    layer.set_output_type(output_index, trt.float32)
                    outputs.append(tensor.name)
            entry = constrained.setdefault(layer.name, {
                "type": str(layer.type),
                "outputs": outputs,
                "layernorm_chains": [],
            })
            entry["layernorm_chains"].append(chain_index)
        chains[chain_index] = chain_names
    return {
        "chain_count": len(chains),
        "chains": chains,
        "constrained_layers": constrained,
    }


def constrain_matching_layers_to_fp32(network, patterns):
    compiled = [re.compile(pattern) for pattern in patterns]
    constrained = {}
    skipped = {}
    producers = {}
    for index in range(network.num_layers):
        producer = network.get_layer(index)
        for output_index in range(producer.num_outputs):
            tensor = producer.get_output(output_index)
            if tensor is not None:
                producers[tensor.name] = (producer, output_index)
    precision_types = {
        trt.LayerType.ACTIVATION,
        trt.LayerType.CONCATENATION,
        trt.LayerType.CONVOLUTION,
        trt.LayerType.ELEMENTWISE,
        trt.LayerType.GATHER,
        trt.LayerType.IDENTITY,
        trt.LayerType.MATRIX_MULTIPLY,
        trt.LayerType.NORMALIZATION,
        trt.LayerType.PLUGIN_V2,
        trt.LayerType.REDUCE,
        trt.LayerType.SCATTER,
        trt.LayerType.SLICE,
        trt.LayerType.SOFTMAX,
        trt.LayerType.UNARY,
        trt.LayerType.SHUFFLE,
    }
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        if not any(pattern.search(layer.name) for pattern in compiled):
            continue
        outputs = [
            layer.get_output(output_index).name
            for output_index in range(layer.num_outputs)
            if layer.get_output(output_index) is not None
            and layer.get_output(output_index).dtype == trt.float32
        ]
        if layer.type not in precision_types or not outputs:
            skipped[layer.name] = {
                "type": str(layer.type),
                "reason": "non-numerical layer or no FP32 output",
            }
            continue
        input_producers = []
        if layer.type == trt.LayerType.PLUGIN_V2:
            for input_index in range(layer.num_inputs):
                tensor = layer.get_input(input_index)
                if tensor is None or tensor.dtype != trt.float32:
                    continue
                producer_entry = producers.get(tensor.name)
                if producer_entry is None:
                    continue
                producer, output_index = producer_entry
                producer.set_output_type(output_index, trt.float32)
                if producer.type in precision_types:
                    producer.precision = trt.float32
                input_producers.append({
                    "tensor": tensor.name,
                    "layer": producer.name,
                    "type": str(producer.type),
                })
        if layer.type != trt.LayerType.PLUGIN_V2:
            layer.precision = trt.float32
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            if tensor is not None and tensor.dtype == trt.float32:
                layer.set_output_type(output_index, trt.float32)
        constrained[layer.name] = {
            "type": str(layer.type),
            "outputs": outputs,
            "input_producers": input_producers,
        }
    return {
        "patterns": patterns,
        "constrained_layers": constrained,
        "skipped_layers": skipped,
    }


def constrain_operator_kinds_to_fp32(network, operator_kinds):
    requested = set(operator_kinds)
    constrained = {}
    skipped = {}
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        operator_kind = None
        if (
            "MatMul" in requested
            and layer.type == trt.LayerType.MATRIX_MULTIPLY
        ):
            operator_kind = "MatMul"
        elif (
            "Mul" in requested
            and layer.type == trt.LayerType.ELEMENTWISE
            and re.search(r"(?:^|/)Mul(?:_|$)", layer.name)
        ):
            operator_kind = "Mul"
        if operator_kind is None:
            continue

        float_outputs = []
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            if tensor is None or tensor.dtype != trt.float32:
                continue
            layer.set_output_type(output_index, trt.float32)
            float_outputs.append(tensor.name)
        if not float_outputs:
            skipped[layer.name] = {
                "operator_kind": operator_kind,
                "type": str(layer.type),
                "reason": "no FP32 output",
            }
            continue
        layer.precision = trt.float32
        constrained[layer.name] = {
            "operator_kind": operator_kind,
            "type": str(layer.type),
            "outputs": float_outputs,
        }
    return {
        "requested_operator_kinds": sorted(requested),
        "mul_match_contract": (
            "TensorRT ELEMENTWISE layer with an ONNX-exported Mul_<id> name"
        ),
        "matched_counts": {
            operator_kind: sum(
                entry["operator_kind"] == operator_kind
                for entry in constrained.values()
            )
            for operator_kind in sorted(requested)
        },
        "constrained_layers": constrained,
        "skipped_layers": skipped,
    }


def constrain_exclusive_output_lineage_to_fp32(network, output_names):
    layers = [network.get_layer(index) for index in range(network.num_layers)]
    layers_by_name = {layer.name: layer for layer in layers}
    producers = {}
    for layer in layers:
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            if tensor is not None:
                producers[tensor.name] = layer.name

    network_outputs = [
        network.get_output(index).name for index in range(network.num_outputs)
    ]
    missing = [name for name in output_names if name not in network_outputs]
    if missing:
        raise RuntimeError(
            "Exclusive FP32 lineage outputs not found in network outputs: %s"
            % missing
        )

    def ancestors(names):
        result = set()
        stack = list(names)
        while stack:
            tensor_name = stack.pop()
            layer_name = producers.get(tensor_name)
            if layer_name is None or layer_name in result:
                continue
            result.add(layer_name)
            layer = layers_by_name[layer_name]
            for input_index in range(layer.num_inputs):
                tensor = layer.get_input(input_index)
                if tensor is not None:
                    stack.append(tensor.name)
        return result

    target_lineage = ancestors(output_names)
    other_outputs = [name for name in network_outputs if name not in output_names]
    other_lineage = ancestors(other_outputs)
    exclusive_lineage = target_lineage - other_lineage
    precision_types = {
        trt.LayerType.ACTIVATION,
        trt.LayerType.CONCATENATION,
        trt.LayerType.CONVOLUTION,
        trt.LayerType.EINSUM,
        trt.LayerType.ELEMENTWISE,
        trt.LayerType.GATHER,
        trt.LayerType.IDENTITY,
        trt.LayerType.MATRIX_MULTIPLY,
        trt.LayerType.NORMALIZATION,
        trt.LayerType.PLUGIN_V2,
        trt.LayerType.REDUCE,
        trt.LayerType.RESIZE,
        trt.LayerType.SCATTER,
        trt.LayerType.SLICE,
        trt.LayerType.SOFTMAX,
        trt.LayerType.UNARY,
    }
    constrained = {}
    skipped = {}
    for layer in layers:
        if layer.name not in exclusive_lineage:
            continue
        float_outputs = []
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            if tensor is not None and tensor.dtype == trt.float32:
                layer.set_output_type(output_index, trt.float32)
                float_outputs.append(tensor.name)
        if not float_outputs:
            skipped[layer.name] = {
                "type": str(layer.type),
                "reason": "no FP32 output",
            }
            continue
        precision_constrained = layer.type in precision_types
        if precision_constrained and layer.type != trt.LayerType.PLUGIN_V2:
            layer.precision = trt.float32
        constrained[layer.name] = {
            "type": str(layer.type),
            "outputs": float_outputs,
            "precision_constrained": precision_constrained,
        }
    return {
        "target_outputs": output_names,
        "other_outputs": other_outputs,
        "target_lineage_layer_count": len(target_lineage),
        "other_lineage_layer_count": len(other_lineage),
        "exclusive_lineage_layer_count": len(exclusive_lineage),
        "constrained_layers": constrained,
        "skipped_layers": skipped,
    }


def constrain_output_lineage_to_fp32(
    network,
    output_names,
    max_depth,
    max_convolutions=0,
    require_network_outputs=True,
    force_plugin_precision=False,
    force_layout_precision=False,
):
    producers = {}
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            if tensor is not None:
                producers[tensor.name] = layer

    available_tensors = producers
    if require_network_outputs:
        available_tensors = {
            network.get_output(index).name: network.get_output(index)
            for index in range(network.num_outputs)
        }
    missing = [name for name in output_names if name not in available_tensors]
    if missing:
        scope = "network outputs" if require_network_outputs else "intermediate tensors"
        raise RuntimeError("FP32 lineage %s not found in %s" % (missing, scope))

    constrained = {}
    visited = set()
    plugin_input_producers = {}
    stack = []
    for name in output_names:
        producer = producers.get(name)
        if producer is None:
            raise RuntimeError("No producer found for network output %s" % name)
        stack.append((producer, 0, name, 0))

    # Keep shape/index/control-flow operations in their native type, while
    # preserving numerical operators and output joins in FP32.
    precision_types = {
        trt.LayerType.ACTIVATION,
        trt.LayerType.CONCATENATION,
        trt.LayerType.ELEMENTWISE,
        trt.LayerType.GATHER,
        trt.LayerType.IDENTITY,
        trt.LayerType.MATRIX_MULTIPLY,
        trt.LayerType.NORMALIZATION,
        trt.LayerType.PLUGIN_V2,
        trt.LayerType.REDUCE,
        trt.LayerType.SCATTER,
        trt.LayerType.SLICE,
        trt.LayerType.SOFTMAX,
        trt.LayerType.UNARY,
    }
    if max_convolutions > 0:
        precision_types.add(trt.LayerType.CONVOLUTION)
    if force_layout_precision:
        precision_types.update({
            trt.LayerType.SHUFFLE,
            trt.LayerType.RESIZE,
            trt.LayerType.SELECT,
        })
    stop_types = {
        trt.LayerType.DEQUANTIZE,
        trt.LayerType.QUANTIZE,
        trt.LayerType.SHAPE,
        trt.LayerType.TOPK,
    }
    while stack:
        layer, depth, root_output, convolution_depth = stack.pop()
        is_convolution = layer.type == trt.LayerType.CONVOLUTION
        current_convolution_depth = convolution_depth + int(is_convolution)
        key = (layer.name, root_output, current_convolution_depth)
        if key in visited or depth > max_depth:
            continue
        visited.add(key)
        float_outputs = []
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            if tensor is not None and tensor.dtype == trt.float32:
                float_outputs.append((output_index, tensor.name))
        for output_index, _ in float_outputs:
            layer.set_output_type(output_index, trt.float32)
        if float_outputs:
            entry = constrained.setdefault(layer.name, {
                "type": str(layer.type),
                "outputs": [name for _, name in float_outputs],
                "root_outputs": [],
                "minimum_depth": depth,
                "minimum_convolution_depth": current_convolution_depth,
                "precision_constrained": layer.type in precision_types,
            })
            if root_output not in entry["root_outputs"]:
                entry["root_outputs"].append(root_output)
            entry["minimum_depth"] = min(entry["minimum_depth"], depth)
            entry["minimum_convolution_depth"] = min(
                entry["minimum_convolution_depth"], current_convolution_depth)
            should_constrain_precision = (
                layer.type in precision_types
                and (layer.type != trt.LayerType.PLUGIN_V2 or force_plugin_precision)
            )
            if should_constrain_precision:
                layer.precision = trt.float32
            if layer.type == trt.LayerType.PLUGIN_V2 and force_plugin_precision:
                # A plugin can expose FP32 outputs while TensorRT still selects
                # a half-precision implementation or half-precision input
                # producer. Keep the numerical plugin boundary explicitly FP32.
                input_entries = []
                for input_index in range(layer.num_inputs):
                    tensor = layer.get_input(input_index)
                    if tensor is None or tensor.dtype != trt.float32:
                        continue
                    producer = producers.get(tensor.name)
                    if producer is None:
                        continue
                    for output_index in range(producer.num_outputs):
                        output = producer.get_output(output_index)
                        if output is not None and output.name == tensor.name:
                            producer.set_output_type(output_index, trt.float32)
                            if producer.type in precision_types:
                                producer.precision = trt.float32
                            input_entries.append({
                                "tensor": tensor.name,
                                "layer": producer.name,
                                "type": str(producer.type),
                            })
                            break
                if input_entries:
                    plugin_input_producers[layer.name] = input_entries
                entry = constrained.get(layer.name)
                if entry is not None:
                    entry["plugin_precision_forced"] = True
                    entry["input_producers"] = input_entries
        if (
            layer.type in stop_types
            or (is_convolution and current_convolution_depth >= max_convolutions)
        ):
            continue
        for input_index in range(layer.num_inputs):
            tensor = layer.get_input(input_index)
            if tensor is None:
                continue
            producer = producers.get(tensor.name)
            if producer is not None:
                stack.append((
                    producer,
                    depth + 1,
                    root_output,
                    current_convolution_depth,
                ))
    return {
        "outputs": output_names,
        "max_depth": max_depth,
        "max_convolutions": max_convolutions,
        "require_network_outputs": require_network_outputs,
        "force_plugin_precision": force_plugin_precision,
        "force_layout_precision": force_layout_precision,
        "plugin_input_producers": plugin_input_producers,
        "constrained_layers": constrained,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--plugin", required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8"), required=True)
    parser.add_argument("--timing-cache", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--workspace-gib", type=int, default=8)
    parser.add_argument("--track-min", type=int, default=901)
    parser.add_argument("--track-opt", type=int, default=960)
    parser.add_argument("--track-max", type=int, default=1300)
    parser.add_argument("--coop-min", type=int, default=1)
    parser.add_argument("--coop-opt", type=int, default=64)
    parser.add_argument("--coop-max", type=int, default=300)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--mark-output", action="append", default=[])
    parser.add_argument("--mark-output-alias", action="append", default=[])
    parser.add_argument("--mark-output-direct-alias", action="append", default=[])
    parser.add_argument("--mark-debug", action="append", default=[])
    parser.add_argument("--outputs-only", action="store_true")
    parser.add_argument("--replace-map-position-layer")
    parser.add_argument("--replace-map-position-input-index", type=int, default=0)
    parser.add_argument("--keep-replaced-map-position-output", action="store_true")
    parser.add_argument("--stabilize-inverse-sigmoid", action="store_true")
    parser.add_argument("--stabilize-layernorm", action="store_true")
    parser.add_argument("--force-fp32-layer-regex", action="append", default=[])
    parser.add_argument(
        "--force-fp32-operator",
        action="append",
        choices=("MatMul", "Mul"),
        default=[],
        help="Force every matching TensorRT numerical operator and its output to FP32.",
    )
    parser.add_argument(
        "--force-fp32-exclusive-output-lineage", action="append", default=[])
    parser.add_argument("--force-fp32-output-lineage", action="append", default=[])
    parser.add_argument("--force-fp32-tensor-lineage", action="append", default=[])
    parser.add_argument("--fp32-lineage-depth", type=int, default=12)
    parser.add_argument("--fp32-lineage-convolutions", type=int, default=0)
    parser.add_argument(
        "--force-fp32-plugin-output-lineage",
        action="store_true",
        help="Also force PluginV2 layers and their FP32 input producers in an output lineage.",
    )
    parser.add_argument(
        "--force-fp32-layout-output-lineage",
        action="store_true",
        help="Also constrain Shuffle/Resize/Select layers in an output lineage.",
    )
    parser.add_argument("--prefer-precision-constraints", action="store_true")
    args = parser.parse_args()
    if args.fp32_lineage_convolutions < 0:
        parser.error("--fp32-lineage-convolutions must be non-negative")

    onnx_path = os.path.realpath(args.onnx)
    plugin_path = os.path.realpath(args.plugin)
    ctypes.CDLL(plugin_path, mode=ctypes.RTLD_GLOBAL)
    logger = CaptureLogger()
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    onnx_parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as handle:
        parsed = onnx_parser.parse(handle.read(), onnx_path)
    parser_errors = [
        str(onnx_parser.get_error(index))
        for index in range(onnx_parser.num_errors)
    ]
    if not parsed or parser_errors:
        raise RuntimeError({"parsed": parsed, "errors": parser_errors})

    map_position_replacement = None
    if args.replace_map_position_layer:
        matching_layers = [
            network.get_layer(index)
            for index in range(network.num_layers)
            if network.get_layer(index).name == args.replace_map_position_layer
        ]
        if len(matching_layers) != 1:
            raise RuntimeError(
                "Expected one TensorRT layer named %s, found %d"
                % (args.replace_map_position_layer, len(matching_layers))
            )
        target_layer = matching_layers[0]
        input_index = args.replace_map_position_input_index
        if input_index < 0 or input_index >= target_layer.num_inputs:
            raise RuntimeError(
                "Invalid input index %d for %s with %d inputs"
                % (input_index, target_layer.name, target_layer.num_inputs)
            )
        replaced_tensor = target_layer.get_input(input_index)
        if replaced_tensor is None:
            raise RuntimeError(
                "Layer %s input %d is empty" % (target_layer.name, input_index)
            )
        expected_shape = (1, 40000, 256)
        replaced_shape = tuple(int(dim) for dim in replaced_tensor.shape)
        if replaced_shape != expected_shape:
            raise RuntimeError(
                "Unexpected map-position shape for %s input %d: %s"
                % (target_layer.name, input_index, replaced_shape)
            )
        map_position_input = network.add_input(
            "map_position_encoding", trt.float32, expected_shape
        )
        if map_position_input is None:
            raise RuntimeError("Failed to add map_position_encoding input")
        target_layer.set_input(input_index, map_position_input)
        if args.keep_replaced_map_position_output:
            network.mark_output(replaced_tensor)
        map_position_replacement = {
            "layer": target_layer.name,
            "input_index": input_index,
            "replaced_tensor": replaced_tensor.name,
            "replaced_shape": list(replaced_shape),
            "replacement_input": map_position_input.name,
            "replacement_shape": list(expected_shape),
            "kept_replaced_tensor_as_output": (
                args.keep_replaced_map_position_output
            ),
        }

    tensors = {}
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        tensors[tensor.name] = tensor
    for index in range(network.num_layers):
        layer = network.get_layer(index)
        for output_index in range(layer.num_outputs):
            tensor = layer.get_output(output_index)
            if tensor is not None:
                tensors[tensor.name] = tensor
    output_aliases = []
    for specification in args.mark_output_alias:
        if "=" not in specification:
            raise ValueError(
                "--mark-output-alias must use SOURCE=ALIAS: %s" % specification
            )
        source_name, alias = specification.split("=", 1)
        if not source_name or not alias:
            raise ValueError(
                "--mark-output-alias must use SOURCE=ALIAS: %s" % specification
            )
        output_aliases.append((source_name, alias))
    direct_output_aliases = []
    for specification in args.mark_output_direct_alias:
        if "=" not in specification:
            raise ValueError(
                "--mark-output-direct-alias must use SOURCE=ALIAS: %s"
                % specification
            )
        source_name, alias = specification.split("=", 1)
        if not source_name or not alias:
            raise ValueError(
                "--mark-output-direct-alias must use SOURCE=ALIAS: %s"
                % specification
            )
        direct_output_aliases.append((source_name, alias))
    requested_tensors = (
        args.mark_output
        + args.mark_debug
        + [source_name for source_name, _ in output_aliases]
        + [source_name for source_name, _ in direct_output_aliases]
    )
    missing_debug_tensors = [name for name in requested_tensors if name not in tensors]
    if missing_debug_tensors:
        raise RuntimeError(
            "TensorRT tensors not found: %s" % missing_debug_tensors
        )
    if args.outputs_only:
        original_outputs = [
            network.get_output(index) for index in range(network.num_outputs)
        ]
        for tensor in original_outputs:
            if tensor.name not in args.mark_output:
                network.unmark_output(tensor)
    marked_names = {
        network.get_output(index).name for index in range(network.num_outputs)
    }
    for name in args.mark_output:
        if name not in marked_names:
            network.mark_output(tensors[name])
            marked_names.add(name)
    output_alias_report = []
    for source_name, alias in output_aliases:
        if alias in tensors or alias in marked_names:
            raise RuntimeError("TensorRT tensor name already exists: %s" % alias)
        identity = network.add_identity(tensors[source_name])
        if identity is None:
            raise RuntimeError("Failed to alias TensorRT tensor %s" % source_name)
        identity.name = "OutputAlias_%s" % alias
        output_tensor = identity.get_output(0)
        output_tensor.name = alias
        network.mark_output(output_tensor)
        marked_names.add(alias)
        output_alias_report.append({"source": source_name, "alias": alias})
    direct_output_alias_report = []
    for source_name, alias in direct_output_aliases:
        if source_name in marked_names:
            raise RuntimeError(
                "Cannot directly alias an existing network output: %s"
                % source_name
            )
        if alias in tensors or alias in marked_names:
            raise RuntimeError("TensorRT tensor name already exists: %s" % alias)
        tensor = tensors[source_name]
        tensor.name = alias
        network.mark_output(tensor)
        marked_names.add(alias)
        direct_output_alias_report.append({
            "source": source_name,
            "alias": alias,
            "direct_network_output": True,
        })
    for name in args.mark_debug:
        network.mark_debug(tensors[name])

    inverse_sigmoid_constraints = None
    if args.stabilize_inverse_sigmoid:
        inverse_sigmoid_constraints = constrain_inverse_sigmoid_to_fp32(network)
        if not inverse_sigmoid_constraints["log_layer_count"]:
            raise RuntimeError("No inverse-sigmoid Log layers found")
    layernorm_constraints = None
    if args.stabilize_layernorm:
        layernorm_constraints = constrain_layernorm_to_fp32(network)
        if not layernorm_constraints["constrained_layers"]:
            raise RuntimeError("No LayerNorm Reduce/Pow layers found")
    regex_constraints = None
    if args.force_fp32_layer_regex:
        regex_constraints = constrain_matching_layers_to_fp32(
            network, args.force_fp32_layer_regex
        )
        if not regex_constraints["constrained_layers"]:
            raise RuntimeError("No layers matched --force-fp32-layer-regex")
    operator_kind_constraints = None
    if args.force_fp32_operator:
        operator_kind_constraints = constrain_operator_kinds_to_fp32(
            network, args.force_fp32_operator
        )
        missing_operator_kinds = [
            operator_kind
            for operator_kind, count
            in operator_kind_constraints["matched_counts"].items()
            if count == 0
        ]
        if missing_operator_kinds:
            raise RuntimeError(
                "No TensorRT layers matched --force-fp32-operator values: %s"
                % missing_operator_kinds
            )
    exclusive_output_lineage_constraints = None
    if args.force_fp32_exclusive_output_lineage:
        exclusive_output_lineage_constraints = (
            constrain_exclusive_output_lineage_to_fp32(
                network, args.force_fp32_exclusive_output_lineage)
        )
        if not exclusive_output_lineage_constraints["constrained_layers"]:
            raise RuntimeError(
                "No layers constrained by "
                "--force-fp32-exclusive-output-lineage"
            )
    output_lineage_constraints = None
    if args.force_fp32_output_lineage:
        output_lineage_constraints = constrain_output_lineage_to_fp32(
            network,
            args.force_fp32_output_lineage,
            args.fp32_lineage_depth,
            args.fp32_lineage_convolutions,
            force_plugin_precision=args.force_fp32_plugin_output_lineage,
            force_layout_precision=args.force_fp32_layout_output_lineage,
        )
        if not output_lineage_constraints["constrained_layers"]:
            raise RuntimeError("No layers constrained by --force-fp32-output-lineage")
    tensor_lineage_constraints = None
    if args.force_fp32_tensor_lineage:
        tensor_lineage_constraints = constrain_output_lineage_to_fp32(
            network,
            args.force_fp32_tensor_lineage,
            args.fp32_lineage_depth,
            args.fp32_lineage_convolutions,
            require_network_outputs=False,
            force_plugin_precision=args.force_fp32_plugin_output_lineage,
            force_layout_precision=args.force_fp32_layout_output_lineage,
        )
        if not tensor_lineage_constraints["constrained_layers"]:
            raise RuntimeError("No layers constrained by --force-fp32-tensor-lineage")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, args.workspace_gib * 1024 ** 3
    )
    config.builder_optimization_level = args.optimization_level
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if not args.allow_tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    if args.precision in ("fp16", "int8"):
        config.set_flag(trt.BuilderFlag.FP16)
    if args.precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
    if (
        inverse_sigmoid_constraints is not None
        or layernorm_constraints is not None
        or regex_constraints is not None
        or operator_kind_constraints is not None
        or exclusive_output_lineage_constraints is not None
        or output_lineage_constraints is not None
        or tensor_lineage_constraints is not None
    ):
        precision_flag = (
            trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS
            if args.prefer_precision_constraints
            else trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS
        )
        config.set_flag(precision_flag)

    profile = builder.create_optimization_profile()
    profile_shapes = {}
    for index in range(network.num_inputs):
        tensor = network.get_input(index)
        shape = tuple(int(dim) for dim in tensor.shape)
        if any(dim < 0 for dim in shape):
            lower, optimum, upper = dynamic_bounds(tensor.name, shape, args)
            accepted = profile.set_shape(tensor.name, lower, optimum, upper)
            if accepted is False:
                raise RuntimeError("Rejected profile for %s" % tensor.name)
            profile_shapes[tensor.name] = {
                "min": list(lower), "opt": list(optimum), "max": list(upper)
            }
    config.add_optimization_profile(profile)

    cache_path = os.path.realpath(args.timing_cache)
    cache_bytes = b""
    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as handle:
            cache_bytes = handle.read()
    initial_cache = {
        "bytes": len(cache_bytes),
        "sha256": hashlib.sha256(cache_bytes).hexdigest()
        if cache_bytes else None,
    }
    timing_cache = config.create_timing_cache(cache_bytes)
    config.set_timing_cache(timing_cache, ignore_mismatch=False)

    start = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_seconds = time.perf_counter() - start
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")

    engine_path = os.path.realpath(args.engine)
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as handle:
        handle.write(serialized)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as handle:
        handle.write(config.get_timing_cache().serialize())

    report = {
        "tensorrt_version": trt.__version__,
        "precision": args.precision,
        "onnx": onnx_path,
        "onnx_sha256": sha256(onnx_path),
        "plugin": plugin_path,
        "engine": engine_path,
        "engine_bytes": os.path.getsize(engine_path),
        "engine_sha256": sha256(engine_path),
        "timing_cache": cache_path,
        "timing_cache_bytes": os.path.getsize(cache_path),
        "build_seconds": build_seconds,
        "workspace_gib": args.workspace_gib,
        "optimization_level": args.optimization_level,
        "tf32_enabled": args.allow_tf32,
        "marked_outputs": args.mark_output,
        "output_aliases": output_alias_report,
        "direct_output_aliases": direct_output_alias_report,
        "marked_debug_tensors": args.mark_debug,
        "outputs_only": args.outputs_only,
        "map_position_replacement": map_position_replacement,
        "inverse_sigmoid_constraints": inverse_sigmoid_constraints,
        "layernorm_constraints": layernorm_constraints,
        "regex_constraints": regex_constraints,
        "operator_kind_constraints": operator_kind_constraints,
        "exclusive_output_lineage_constraints": (
            exclusive_output_lineage_constraints),
        "output_lineage_constraints": output_lineage_constraints,
        "tensor_lineage_constraints": tensor_lineage_constraints,
        "prefer_precision_constraints": args.prefer_precision_constraints,
        "force_fp32_plugin_output_lineage": args.force_fp32_plugin_output_lineage,
        "force_fp32_layout_output_lineage": args.force_fp32_layout_output_lineage,
        "initial_timing_cache": initial_cache,
        "network_layers": int(network.num_layers),
        "inputs": [tensor_metadata(network.get_input(index)) for index in range(network.num_inputs)],
        "outputs": [tensor_metadata(network.get_output(index)) for index in range(network.num_outputs)],
        "profile_shapes": profile_shapes,
        "logger_messages": logger.messages,
    }
    report_path = os.path.realpath(args.report)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({
        "engine": engine_path,
        "engine_bytes": report["engine_bytes"],
        "build_seconds": build_seconds,
        "network_layers": report["network_layers"],
        "profile_shapes": profile_shapes,
        "report": report_path,
    }, indent=2))


if __name__ == "__main__":
    main()
