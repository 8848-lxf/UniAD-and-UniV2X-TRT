import argparse
import hashlib
import json
import os

import onnx


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def external_files(path):
    model = onnx.load(path, load_external_data=False)
    locations = set()
    for initializer in model.graph.initializer:
        for entry in initializer.external_data:
            if entry.key == "location":
                locations.add(entry.value)
    result = []
    for location in sorted(locations):
        source = os.path.join(os.path.dirname(os.path.abspath(path)), location)
        result.append({
            "path": source,
            "bytes": os.path.getsize(source),
            "sha256": sha256(source),
        })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    sources = external_files(input_path)
    model = onnx.load(input_path, load_external_data=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    onnx.save(model, output_path)
    reloaded = onnx.load(output_path, load_external_data=False)
    if any(initializer.external_data for initializer in reloaded.graph.initializer):
        raise RuntimeError("Materialized ONNX still contains external data")

    report = {
        "input": {
            "path": input_path,
            "bytes": os.path.getsize(input_path),
            "sha256": sha256(input_path),
            "external_data": sources,
        },
        "output": {
            "path": output_path,
            "bytes": os.path.getsize(output_path),
            "sha256": sha256(output_path),
            "external_data": [],
        },
        "graph": {
            "nodes": len(reloaded.graph.node),
            "initializers": len(reloaded.graph.initializer),
            "inputs": len(reloaded.graph.input),
            "outputs": len(reloaded.graph.output),
        },
    }
    report_path = os.path.abspath(args.report)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
