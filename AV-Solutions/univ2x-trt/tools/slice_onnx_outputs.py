import argparse
import json
import os

import onnx
import onnx_graphsurgeon as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--report")
    args = parser.parse_args()

    model = onnx.load(args.input, load_external_data=True)
    graph = gs.import_onnx(model)
    tensors = graph.tensors()
    missing = [name for name in args.outputs if name not in tensors]
    if missing:
        raise KeyError(f"Unknown graph outputs: {missing}")
    graph.outputs = [tensors[name] for name in args.outputs]
    graph.cleanup(remove_unused_graph_inputs=True).toposort()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    onnx.save_model(
        gs.export_onnx(graph),
        args.output,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=os.path.basename(args.output) + ".data",
        size_threshold=1024,
    )
    result = {
        "source": os.path.abspath(args.input),
        "output": os.path.abspath(args.output),
        "outputs": args.outputs,
        "nodes": len(graph.nodes),
        "inputs": [value.name for value in graph.inputs],
        "onnx_bytes": os.path.getsize(args.output),
        "external_data_bytes": os.path.getsize(args.output + ".data"),
    }
    if args.report:
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w") as handle:
            json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
