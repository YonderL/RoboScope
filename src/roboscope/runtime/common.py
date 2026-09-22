"""Worker utilities; shared by ACT and DP."""

import json
import random
from pathlib import Path

import numpy as np
import torch

from roboscope.data.libero import digest as digest
from roboscope.data.libero import save_json as save_json

CAMERAS = ("agentview_rgb", "eye_in_hand_rgb")


def load_config(path):
    return json.loads(Path(path).read_text())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_4090():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Worker requires exactly one UUID-masked CUDA device")
    if "4090" not in torch.cuda.get_device_name(0):
        raise RuntimeError("Refusing a non-4090 GPU")


def atomic_save(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    tmp.replace(path)


def variants(cfg):
    if cfg.get("policy") == "smolvla_rlt":
        label = "reference" if cfg.get("evaluation_policy") == "sft_reference" else "rlt"
        return [
            {
                "name": f"smolvla_{label}_c{cfg['action_horizon']:03d}",
                "model": "rlt",
                "ta": cfg["action_horizon"],
                "ddim_steps": 0,
            }
        ]
    if cfg.get("policy") == "smolvla":
        return [{"name": "smolvla_chunk50", "model": "smolvla", "ta": 50, "ddim_steps": 0}]
    pairs = {(n, cfg["action_horizon"]) for n in cfg["ddim_ablation"]}
    pairs |= {(cfg["ddim_steps"], ta) for ta in cfg["action_horizon_ablation"]}
    return [
        {"name": f"dp_ddim{n:02d}_ta{ta:02d}", "model": "dp", "ddim_steps": n, "ta": ta}
        for n, ta in sorted(pairs)
    ] + [{"name": "act_k008_chunk", "model": "act", "ddim_steps": 0, "ta": 8}]
