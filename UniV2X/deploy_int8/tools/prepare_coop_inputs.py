import argparse
import json
import os

import numpy as np
import torch

import projects.mmdet3d_plugin  # noqa: F401

from trt_runtime import (
    COOP_INPUT_NAMES,
    INFRASTRUCTURE_OUTPUT_NAMES,
    build_trt_agent,
    load_export_inputs,
)


INFRASTRUCTURE_PC_RANGE = [0.0, -51.2, -5.0, 102.4, 51.2, 3.0]


def denormalize(ref_pts, pc_range):
    scale = ref_pts.new_tensor([
        pc_range[3] - pc_range[0],
        pc_range[4] - pc_range[1],
        pc_range[5] - pc_range[2],
    ])
    offset = ref_pts.new_tensor(pc_range[:3])
    return ref_pts.sigmoid() * scale + offset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("infrastructure_npz")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    _, model = build_trt_agent(
        args.config, args.checkpoint, "infrastructure"
    )
    model = model.cuda().eval()
    inputs = load_export_inputs(
        args.infrastructure_npz, torch.device("cuda")
    )
    with torch.no_grad():
        outputs = model.forward_uniad_trt(*inputs)
    output_map = dict(zip(INFRASTRUCTURE_OUTPUT_NAMES, outputs))
    tracks = [output_map[f"prev_track_intances{index}_out"] for index in range(14)]

    active = tracks[3] >= 0
    tracks = [value[active] for value in tracks]
    source = np.load(args.infrastructure_npz)
    veh2inf_rt = source["veh2inf_rt"]
    other2ego = np.linalg.inv(veh2inf_rt[0].T).astype(np.float32)[None]
    other2ego_tensor = tracks[1].new_tensor(other2ego)
    locs = denormalize(tracks[1], INFRASTRUCTURE_PC_RANGE)
    homogeneous = torch.cat(
        [locs, torch.ones_like(locs[:, :1])], dim=-1
    )
    ego_locs = torch.matmul(
        other2ego_tensor[0], homogeneous.unsqueeze(-1)
    ).squeeze(-1)[:, :3]
    in_ego_box = (
        (ego_locs[:, 0] >= -2.04)
        & (ego_locs[:, 0] <= 2.04)
        & (ego_locs[:, 1] >= -0.92)
        & (ego_locs[:, 1] <= 0.92)
    )
    keep = torch.ones_like(in_ego_box)
    ego_candidates = torch.where(in_ego_box)[0]
    if ego_candidates.numel() > 0:
        keep[ego_candidates[0]] = False
    tracks = [value[keep] for value in tracks]

    values = {
        f"coop_track_intances{index}": value.detach().cpu().numpy()
        for index, value in enumerate(tracks)
    }
    # The first ego frame contains no active vehicle tracks, so all valid
    # infrastructure tracks are unmatched and are appended by query fusion.
    values["coop_match_vehicle_index"] = np.full(
        (tracks[0].shape[0],), -1, dtype=np.int64
    )
    values["coop_other2ego_rt"] = other2ego
    values["coop_lane_outputs_classes"] = output_map[
        "lane_outputs_classes"
    ].detach().cpu().numpy()
    values["coop_lane_outputs_coords"] = output_map[
        "lane_outputs_coords"
    ].detach().cpu().numpy()
    values["coop_lane_query"] = output_map["lane_query"].detach().cpu().numpy()
    values["coop_lane_query_pos"] = output_map[
        "lane_query_pos"
    ].detach().cpu().numpy()
    values["coop_lane_reference"] = output_map[
        "lane_reference"
    ].detach().cpu().numpy()
    values["coop_occ_pred_scores"] = output_map[
        "occ_pred_scores"
    ].detach().cpu().numpy()
    values["coop_ego2other_rt"] = veh2inf_rt.astype(np.float32)
    if list(values) != COOP_INPUT_NAMES:
        raise AssertionError((list(values), COOP_INPUT_NAMES))

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez(args.output, **values)
    report = {
        "source": os.path.abspath(args.infrastructure_npz),
        "output": os.path.abspath(args.output),
        "active_before_ego_removal": int(active.sum().item()),
        "ego_instances_removed": int((~keep).sum().item()),
        "cooperative_track_count": int(tracks[0].shape[0]),
        "tensors": {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in values.items()
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
