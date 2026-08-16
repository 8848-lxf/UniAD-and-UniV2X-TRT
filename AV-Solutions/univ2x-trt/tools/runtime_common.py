import importlib
import os

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model


def import_plugin(cfg):
    if not cfg.get("plugin", False):
        return
    plugin_dir = cfg.get("plugin_dir") or os.path.dirname(cfg.filename)
    module_path = ".".join(os.path.dirname(plugin_dir).split("/"))
    importlib.import_module(module_path)


def load_config(config_path, workers=0):
    cfg = Config.fromfile(config_path)
    import_plugin(cfg)
    cfg.data.workers_per_gpu = workers
    cfg.data.test.test_mode = True
    cfg.model_ego_agent.pretrained = None
    return cfg


def build_dataset_and_model(config_path, checkpoint_path, workers=0):
    from projects.mmdet3d_plugin.univ2x.detectors.multi_agent import MultiAgent

    cfg = load_config(config_path, workers=workers)
    dataset = build_dataset(cfg.data.test)

    other_agent_models = {}
    for name in cfg.keys():
        if "model_other_agent" not in name:
            continue
        agent_cfg = cfg.get(name)
        agent_cfg.train_cfg = None
        other_agent_models[name] = build_model(
            agent_cfg, test_cfg=cfg.get("test_cfg")
        )

    cfg.model_ego_agent.train_cfg = None
    ego_model = build_model(
        cfg.model_ego_agent, test_cfg=cfg.get("test_cfg")
    )
    model = MultiAgent(ego_model, other_agent_models)
    checkpoint = load_checkpoint(model, checkpoint_path, map_location="cpu")

    classes = checkpoint.get("meta", {}).get("CLASSES", dataset.CLASSES)
    model.model_ego_agent.CLASSES = classes
    if hasattr(dataset, "PALETTE"):
        model.model_ego_agent.PALETTE = checkpoint.get("meta", {}).get(
            "PALETTE", dataset.PALETTE
        )
    return cfg, dataset, model, checkpoint
