import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEPLOY_ROOT = os.path.join(REPO_ROOT, "deploy_int8", "UniV2X_deploy")
TRT_FUNCTIONS = os.path.join(
    DEPLOY_ROOT, "projects", "mmdet3d_plugin", "uniad", "functions"
)
for source_root in (REPO_ROOT, DEPLOY_ROOT, TRT_FUNCTIONS):
    while source_root in sys.path:
        sys.path.remove(source_root)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, TRT_FUNCTIONS)
sys.path.insert(0, DEPLOY_ROOT)

import projects.mmdet3d_plugin  # noqa: E402,F401

from trt_runtime import (  # noqa: E402
    EGO_INPUT_NAMES,
    INFRASTRUCTURE_OUTPUT_NAMES,
    INPUT_NAMES,
    OUTPUT_NAMES,
    build_trt_agent,
)


def compare(reference, actual):
    result = {
        "shape": list(actual.shape),
        "shape_match": tuple(reference.shape) == tuple(actual.shape),
    }
    if not result["shape_match"]:
        result["reference_shape"] = list(reference.shape)
        return result
    if actual.is_floating_point():
        difference = (reference.float() - actual.float()).abs()
        result.update({
            "max_abs_error": float(difference.max().item()),
            "mean_abs_error": float(difference.mean().item()),
            "allclose_rtol_1e-3_atol_1e-4": bool(torch.allclose(
                reference.float(), actual.float(), rtol=1.0e-3, atol=1.0e-4
            )),
        })
    else:
        result["mismatch_count"] = int((reference != actual).sum().item())
        result["element_count"] = int(actual.numel())
        result["mismatch_rate"] = (
            result["mismatch_count"] / result["element_count"]
            if result["element_count"] else 0.0
        )
        result["reference_nonzero"] = int(torch.count_nonzero(reference).item())
        result["actual_nonzero"] = int(torch.count_nonzero(actual).item())
    return result


def compare_lane_segments(reference, actual, local_query_count=300):
    if tuple(reference.shape) != tuple(actual.shape):
        return None
    if reference.ndim == 3:
        query_axis = 1
    elif reference.ndim == 4:
        query_axis = 2
    else:
        return None
    query_count = reference.shape[query_axis]
    if query_count <= local_query_count:
        return None
    local_slice = [slice(None)] * reference.ndim
    cooperative_slice = [slice(None)] * reference.ndim
    local_slice[query_axis] = slice(0, local_query_count)
    cooperative_slice[query_axis] = slice(local_query_count, None)
    local_slice = tuple(local_slice)
    cooperative_slice = tuple(cooperative_slice)
    return {
        "local_query_count": local_query_count,
        "cooperative_query_count": query_count - local_query_count,
        "local": compare(reference[local_slice], actual[local_slice]),
        "cooperative": compare(
            reference[cooperative_slice], actual[cooperative_slice]
        ),
    }


def compare_box_sets(
    reference_boxes,
    actual_boxes,
    reference_scores,
    actual_scores,
    reference_labels,
    actual_labels,
):
    if reference_boxes.ndim != 2 or actual_boxes.ndim != 2:
        return None
    if reference_boxes.shape != actual_boxes.shape or reference_boxes.shape[1] < 3:
        return None
    if reference_boxes.shape[0] == 0:
        return {"box_count": 0, "label_mismatches": 0}
    reference_centers = reference_boxes[:, :3].detach().float().cpu()
    actual_centers = actual_boxes[:, :3].detach().float().cpu()
    reference_labels_cpu = reference_labels.detach().long().cpu()
    actual_labels_cpu = actual_labels.detach().long().cpu()
    cost = torch.cdist(reference_centers, actual_centers)
    cost += (
        reference_labels_cpu[:, None] != actual_labels_cpu[None, :]
    ).float() * 1.0e6
    reference_index, actual_index = linear_sum_assignment(cost.numpy())
    reference_index = torch.from_numpy(reference_index).long()
    actual_index = torch.from_numpy(actual_index).long()
    matched_boxes = actual_boxes.detach().float().cpu()[actual_index]
    selected_reference_boxes = reference_boxes.detach().float().cpu()[reference_index]
    box_difference = (selected_reference_boxes - matched_boxes).abs()
    matched_labels = actual_labels_cpu[actual_index]
    selected_reference_labels = reference_labels_cpu[reference_index]
    matched_scores = actual_scores.detach().float().cpu()[actual_index]
    selected_reference_scores = reference_scores.detach().float().cpu()[reference_index]
    score_difference = (selected_reference_scores - matched_scores).abs()
    return {
        "box_count": int(reference_boxes.shape[0]),
        "label_mismatches": int(
            (selected_reference_labels != matched_labels).sum().item()
        ),
        "box_mean_abs_error": float(box_difference.mean().item()),
        "box_max_abs_error": float(box_difference.max().item()),
        "center_mean_l2_error": float(
            torch.linalg.vector_norm(
                selected_reference_boxes[:, :3] - matched_boxes[:, :3], dim=1
            ).mean().item()
        ),
        "score_mean_abs_error": float(score_difference.mean().item()),
        "score_max_abs_error": float(score_difference.max().item()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("snapshot_glob")
    parser.add_argument("--agent", choices=("ego", "infrastructure"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--agent-normalization-epsilon", type=float, default=0.0009765625
    )
    args = parser.parse_args()

    input_names = EGO_INPUT_NAMES if args.agent == "ego" else INPUT_NAMES
    output_names = OUTPUT_NAMES if args.agent == "ego" else INFRASTRUCTURE_OUTPUT_NAMES
    _, model = build_trt_agent(args.config, args.checkpoint, args.agent)
    if args.agent == "ego":
        model.cross_agent_query_interaction.normalization_epsilon = (
            args.agent_normalization_epsilon
        )
    model = model.cuda().eval()

    reports = []
    for path in sorted(glob.glob(args.snapshot_glob)):
        archive = np.load(path)
        inputs = tuple(
            torch.from_numpy(archive[f"input::{name}"]).cuda()
            for name in input_names
        )
        with torch.no_grad():
            reference_values = model.forward_uniad_trt(*inputs)
        reference = dict(zip(output_names, reference_values))
        actual = {
            key.split("::", 1)[1]: torch.from_numpy(archive[key]).cuda()
            for key in archive.files if key.startswith("output::")
        }
        common = sorted(set(reference) & set(actual))
        comparisons = {
            name: compare(reference[name], actual[name]) for name in common
        }
        lane_segments = {}
        for name in (
            "lane_outputs_classes",
            "lane_outputs_coords",
            "lane_query",
            "lane_query_pos",
            "lane_reference",
        ):
            if name not in common:
                continue
            segmented = compare_lane_segments(reference[name], actual[name])
            if segmented is not None:
                lane_segments[name] = segmented
        detection_set_comparisons = {}
        for prefix, names in {
            "tracking": (
                "bboxes_dict_bboxes", "scores", "labels"
            ),
            "detection": (
                "det_bboxes", "det_scores", "det_labels"
            ),
        }.items():
            boxes_name, scores_name, labels_name = names
            if not all(name in common for name in names):
                continue
            detection_set_comparisons[prefix] = compare_box_sets(
                reference[boxes_name],
                actual[boxes_name],
                reference[scores_name],
                actual[scores_name],
                reference[labels_name],
                actual[labels_name],
            )
        reports.append({
            "snapshot": os.path.abspath(path),
            "outputs": comparisons,
            "lane_query_segments": lane_segments,
            "detection_set_comparisons": detection_set_comparisons,
            "shape_mismatches": [
                name for name, value in comparisons.items()
                if not value["shape_match"]
            ],
            "integer_mismatches": {
                name: value["mismatch_count"]
                for name, value in comparisons.items()
                if value.get("mismatch_count", 0)
            },
            "floating_not_allclose": [
                name for name, value in comparisons.items()
                if value.get("allclose_rtol_1e-3_atol_1e-4") is False
            ],
        })

    result = {
        "agent": args.agent,
        "agent_normalization_epsilon": (
            args.agent_normalization_epsilon if args.agent == "ego" else None
        ),
        "gpu": torch.cuda.get_device_name(0),
        "snapshots": reports,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({
        "agent": args.agent,
        "snapshots": len(reports),
        "summary": [
            {
                "snapshot": os.path.basename(item["snapshot"]),
                "shape_mismatches": item["shape_mismatches"],
                "integer_mismatches": item["integer_mismatches"],
                "floating_not_allclose": item["floating_not_allclose"],
            }
            for item in reports
        ],
        "output": os.path.abspath(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
