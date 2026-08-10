import argparse
import ctypes
import json
import os

import tensorrt as trt


class CaptureLogger(trt.ILogger):
    def __init__(self, severity=trt.ILogger.Severity.WARNING):
        super().__init__()
        self.min_severity = severity
        self.messages = []

    def log(self, severity, message):
        self.messages.append({
            "severity": str(severity).rsplit(".", 1)[-1],
            "message": message,
        })
        if severity <= self.min_severity:
            print("[TensorRT %s] %s" % (severity, message))


def tensor_metadata(tensor):
    return {
        "name": tensor.name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("onnx")
    parser.add_argument("--plugin", action="append", default=[])
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    loaded_plugins = []
    for path in args.plugin:
        resolved = os.path.realpath(path)
        ctypes.CDLL(resolved, mode=ctypes.RTLD_GLOBAL)
        loaded_plugins.append(resolved)

    logger = CaptureLogger()
    trt.init_libnvinfer_plugins(logger, "")
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    onnx_parser = trt.OnnxParser(network, logger)
    onnx_path = os.path.realpath(args.onnx)
    with open(onnx_path, "rb") as handle:
        parsed = onnx_parser.parse(handle.read(), onnx_path)

    parser_errors = [str(onnx_parser.get_error(index))
                     for index in range(onnx_parser.num_errors)]
    report = {
        "tensorrt_version": trt.__version__,
        "onnx": onnx_path,
        "plugins": loaded_plugins,
        "parsed": bool(parsed),
        "parser_errors": parser_errors,
        "network_layers": int(network.num_layers),
        "inputs": [tensor_metadata(network.get_input(index))
                   for index in range(network.num_inputs)],
        "outputs": [tensor_metadata(network.get_output(index))
                    for index in range(network.num_outputs)],
        "logger_messages": logger.messages,
    }
    report_path = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({
        "parsed": report["parsed"],
        "parser_error_count": len(parser_errors),
        "network_layers": report["network_layers"],
        "input_count": len(report["inputs"]),
        "output_count": len(report["outputs"]),
        "report": report_path,
    }, indent=2))
    if not parsed or parser_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
