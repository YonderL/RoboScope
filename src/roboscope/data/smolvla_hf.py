"""HF Spatial token-training observations and a pinned native SFT/RLT contract."""

import json
import tempfile
from pathlib import Path

import torch
from torch.utils.data import Dataset

from roboscope.data.smolvla_official import DATASET_ID, DATASET_REVISION, validate_features
from roboscope.policies.smolvla_hf import checkpoint_files, validate_config
from roboscope.runtime.common import digest


def prepare_contract(source, cfg, checkpoint="final"):
    """Read-only preflight: never translate old joint-state checkpoints into HF runs."""
    from safetensors.torch import load_file

    from roboscope.envs.hf_libero import setup

    source = Path(source).resolve()
    saved = json.loads((source / "config.json").read_text())
    if saved.get("protocol") != "smolvla_paper_450m_spatial_only":
        raise ValueError("HF RLT requires the official HF Spatial SmolVLA SFT run")
    if checkpoint != "final":
        raise ValueError("Native HF SFT uses the final checkpoint; validation-best is not defined")
    weight = (source / "training/checkpoints/last/pretrained_model").resolve()
    step_file = weight.parent / "training_state/training_step.json"
    if json.loads(step_file.read_text())["step"] != saved["steps"]:
        raise ValueError("The native last checkpoint has not reached the configured final SFT step")
    config = json.loads((weight / "config.json").read_text())
    validate_config(config)
    dataset = Path(saved["dataset_root"]).resolve()
    provenance = json.loads((dataset / "spatial_provenance.json").read_text())
    if (provenance["source_repo"], provenance["source_revision"]) != (DATASET_ID, DATASET_REVISION):
        raise ValueError("HF Spatial dataset must use the pinned, verified source revision")
    if provenance != json.loads((source / "manifest.json").read_text()):
        raise ValueError("HF Spatial provenance differs from the frozen SFT run")
    if saved["dataset_repo"] != DATASET_ID or saved["dataset_revision"] != DATASET_REVISION:
        raise ValueError("SFT dataset revision differs from HF RLT")
    validate_features(json.loads((dataset / "meta/info.json").read_text()))
    files = checkpoint_files(weight)
    preprocessor = json.loads((weight / "policy_preprocessor.json").read_text())
    normalizer = next(s for s in preprocessor["steps"] if s["registry_name"] == "normalizer_processor")
    if normalizer["config"]["norm_map"]["STATE"] != "MEAN_STD":
        raise ValueError("RLT proprioception requires the checkpoint's mean/std normalization")
    stats = load_file(str(weight / normalizer["state_file"]))
    effective = {
        **cfg,
        "sft_source": str(source),
        "sft_checkpoint": str(weight),
        "sft_sha256": files["model.safetensors"],
        "sft_files": files,
        "sft_source_files": {name: digest(source / name) for name in ("config.json", "manifest.json")},
        "data_root": str(dataset),
        "dataset_root": str(dataset),
        "libero_root": saved["libero_root"],
        "state_dim": 8,
        "sft_step": saved["steps"],
    }
    with tempfile.TemporaryDirectory(prefix="roboscope-hf-contract-") as folder:
        suite = setup(effective, Path(folder))
        tasks = []
        root = Path(saved["libero_root"])
        for i in range(suite.n_tasks):
            task = suite.get_task(i)
            bddl = root / "bddl_files" / task.problem_folder / task.bddl_file
            initial = root / "init_files" / task.problem_folder / task.init_states_file
            tasks.append(
                {
                    "id": i,
                    "name": task.name,
                    "language": task.language,
                    "bddl": str(bddl),
                    "init": str(initial),
                    "bddl_sha256": digest(bddl),
                    "init_sha256": digest(initial),
                    "eval_initial_state_ids": list(range(cfg["eval_episodes"])),
                }
            )
    if len(tasks) != 10 or {t["language"] for t in tasks} != set(provenance["tasks"]):
        raise ValueError("Native benchmark languages differ from the HF Spatial demonstrations")
    manifest = {
        "schema": "roboscope.smolvla_hf_rlt.v1",
        "dataset_format": "hf_libero_spatial",
        "dataset_root": str(dataset),
        "dataset_repo": f"{DATASET_ID}_spatial",
        "dataset_provenance": provenance,
        "dataset_metadata_sha256": {
            str(p.relative_to(dataset)): digest(p)
            for p in sorted((dataset / "meta").rglob("*"))
            if p.is_file()
        },
        "tasks": tasks,
        "state_mean": stats["observation.state.mean"].tolist(),
        "state_std": stats["observation.state.std"].tolist(),
        "state_eps": normalizer["config"]["eps"],
        "sft_files": files,
        "token_demonstrations": "all SFT demonstration episodes; SFT eval_split=0",
    }
    return effective, manifest


def verify_contract(cfg, manifest):
    """Protect both inference and resume against swapped weights/processors/data/assets."""
    from roboscope.envs.hf_libero import check_runtime

    check_runtime(cfg)
    if (
        checkpoint_files(cfg["sft_checkpoint"]) != cfg["sft_files"]
        or manifest["sft_files"] != cfg["sft_files"]
    ):
        raise ValueError("Frozen HF SFT checkpoint or processor files changed")
    for name, expected in cfg["sft_source_files"].items():
        if digest(Path(cfg["sft_source"]) / name) != expected:
            raise ValueError("Frozen SFT source contract changed")
    root = Path(cfg["dataset_root"])
    if json.loads((root / "spatial_provenance.json").read_text()) != manifest["dataset_provenance"]:
        raise ValueError("HF dataset provenance changed")
    for name, expected in manifest["dataset_metadata_sha256"].items():
        if digest(root / name) != expected:
            raise ValueError("HF dataset metadata changed")
    for task in manifest["tasks"]:
        for key in ("bddl", "init"):
            if digest(task[key]) != task[f"{key}_sha256"]:
                raise ValueError("HF LIBERO benchmark assets changed")


class HFSpatialFrames(Dataset):
    """Read the same local LeRobot release as native SFT; no resize, flip or new split."""

    def __init__(self, manifest):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.dataset = LeRobotDataset(manifest["dataset_repo"], root=manifest["dataset_root"])
        self.task_ids = {task["language"]: task["id"] for task in manifest["tasks"]}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        language = sample["task"]
        # Dataset task indices are lexicographic; simulator IDs use native suite order.
        result = {"state": sample["observation.state"], "task_id": torch.tensor(self.task_ids[language])}
        for source, target in (("image", "agentview_rgb"), ("image2", "eye_in_hand_rgb")):
            result[target] = (sample[f"observation.images.{source}"] * 255).round().to(torch.uint8)
        return result
