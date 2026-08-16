import json
import os
import os.path as osp
import types

import numpy as np
import torch


def tensor_summary(value, sample_count=2048):
    tensor = value.detach().float().cpu().contiguous()
    flat = tensor.view(-1)
    finite = torch.isfinite(flat)
    finite_values = flat[finite]
    if flat.numel() <= sample_count:
        sample_indices = torch.arange(flat.numel(), dtype=torch.long)
    else:
        sample_indices = torch.linspace(
            0, flat.numel() - 1, sample_count, dtype=torch.float64
        ).long()
    samples = flat[sample_indices]
    return {
        "shape": list(tensor.shape),
        "elements": int(flat.numel()),
        "nonfinite": int((~finite).sum().item()),
        "min": float(finite_values.min().item()) if finite_values.numel() else None,
        "max": float(finite_values.max().item()) if finite_values.numel() else None,
        "mean": float(finite_values.mean().item()) if finite_values.numel() else None,
        "std": float(finite_values.std(unbiased=False).item())
        if finite_values.numel()
        else None,
        "l2": float(torch.linalg.vector_norm(finite_values).item())
        if finite_values.numel()
        else None,
        "sample_indices": sample_indices.tolist(),
        "samples": [
            float(item) if torch.isfinite(item) else None for item in samples
        ],
    }


class OccInputTrace:
    def __init__(self, output_dir, frames):
        self.output_dir = output_dir
        self.frames = set(frames)
        self.current_frame = None
        self.current = {}
        os.makedirs(output_dir, exist_ok=True)

    def set_frame(self, frame_index):
        self.current_frame = frame_index
        self.current = {}

    def wrap(self, head, agent, method_name):
        original = getattr(head, method_name)

        def traced(module, bev_feat, ins_query):
            output = original(bev_feat, ins_query)
            if self.current_frame in self.frames:
                logits = output[-1] if isinstance(output, (tuple, list)) else output
                self.current.setdefault(agent, {}).update({
                    "bev_feat": tensor_summary(bev_feat),
                    "ins_query": tensor_summary(ins_query),
                    "occ_logits": tensor_summary(logits),
                })
            return output

        setattr(head, method_name, types.MethodType(traced, head))

    @staticmethod
    def _summarize_structure(value, prefix, result, limit=96):
        if len(result) >= limit:
            return
        if isinstance(value, torch.Tensor):
            result[prefix] = tensor_summary(value)
            return
        if hasattr(value, "get_fields"):
            OccInputTrace._summarize_structure(
                value.get_fields(), prefix, result, limit
            )
            return
        if isinstance(value, dict):
            for key, item in value.items():
                OccInputTrace._summarize_structure(
                    item, prefix + "." + str(key), result, limit
                )
            return
        if isinstance(value, (tuple, list)):
            for index, item in enumerate(value):
                OccInputTrace._summarize_structure(
                    item, prefix + "." + str(index), result, limit
                )

    def wrap_module(self, module, agent, stage):
        def hook(_module, inputs, output):
            if self.current_frame not in self.frames:
                return
            values = {}
            self._summarize_structure(inputs, "input", values)
            self._summarize_structure(output, "output", values)
            self.current.setdefault(agent, {})[stage] = values

        module.register_forward_hook(hook)

    def wrap_structure_method(self, module, agent, stage, method_name):
        original = getattr(module, method_name)

        def traced(_module, *args, **kwargs):
            output = original(*args, **kwargs)
            if self.current_frame in self.frames:
                values = {}
                self._summarize_structure(args, "input", values)
                self._summarize_structure(kwargs, "keyword", values)
                self._summarize_structure(output, "output", values)
                self.current.setdefault(agent, {})[stage] = values
            return output

        setattr(module, method_name, types.MethodType(traced, module))

    def wrap_common_encoder_stages(self, model, agent, deep=False):
        self.wrap_module(model.img_backbone, agent, "img_backbone")
        self.wrap_module(model.img_neck, agent, "img_neck")
        self.wrap_module(
            model.pts_bbox_head.positional_encoding,
            agent,
            "bev_positional_encoding",
        )
        transformer = model.pts_bbox_head.transformer
        self.wrap_module(transformer.can_bus_mlp, agent, "can_bus_mlp")
        encoder = transformer.encoder
        if deep:
            if hasattr(encoder, "point_sampling_trt"):
                self.wrap_structure_method(
                    encoder, agent, "point_sampling", "point_sampling_trt"
                )
            else:
                self.wrap_structure_method(
                    encoder, agent, "point_sampling", "point_sampling"
                )
            for layer_index, layer in enumerate(encoder.layers):
                if hasattr(layer, "forward_trt"):
                    self.wrap_structure_method(
                        layer,
                        agent,
                        "bev_encoder_layer_%d" % layer_index,
                        "forward_trt",
                    )
                else:
                    self.wrap_module(
                        layer, agent, "bev_encoder_layer_%d" % layer_index
                    )
                for attention_index, attention in enumerate(layer.attentions):
                    stage = "bev_encoder_layer_%d_attention_%d" % (
                        layer_index,
                        attention_index,
                    )
                    if hasattr(attention, "forward_trt"):
                        self.wrap_structure_method(
                            attention, agent, stage, "forward_trt"
                        )
                    else:
                        self.wrap_module(attention, agent, stage)
        if hasattr(encoder, "forward_trt"):
            self.wrap_structure_method(
                encoder, agent, "bev_encoder", "forward_trt"
            )
        else:
            self.wrap_module(encoder, agent, "bev_encoder")

    def wrap_cooperative_stages(self, model, agent):
        fusion = model.cross_agent_query_interaction
        if hasattr(fusion, "forward_trt"):
            self.wrap_structure_method(
                fusion, agent, "agent_query_fusion", "forward_trt"
            )
            self.wrap_structure_method(
                model, agent, "track_bev_fusion", "augment_track_bev_trt"
            )
        else:
            self.wrap_module(fusion, agent, "agent_query_fusion")
            self.wrap_structure_method(
                model, agent, "track_bev_fusion", "_get_coop_bev_embed"
            )

    def wrap_motion_stage(self, model, agent):
        if hasattr(model.motion_head, "forward_test_trt"):
            self.wrap_structure_method(
                model.motion_head, agent, "motion_head", "forward_test_trt"
            )
        else:
            self.wrap_structure_method(
                model.motion_head, agent, "motion_head", "forward_test"
            )

    def wrap_detection_stage(self, model, agent):
        head = model.pts_bbox_head
        if hasattr(head, "get_detections_trt"):
            self.wrap_structure_method(
                head, agent, "detection_head", "get_detections_trt"
            )
        else:
            self.wrap_structure_method(
                head, agent, "detection_head", "get_detections"
            )

    def flush(self):
        if self.current_frame not in self.frames:
            return
        path = osp.join(self.output_dir, "frame_%04d.json" % self.current_frame)
        temporary = path + ".tmp"
        with open(temporary, "w") as handle:
            json.dump(self.current, handle, allow_nan=False)
        os.replace(temporary, path)


def parse_frames(value):
    return [int(item) for item in value.split(",") if item.strip()]
