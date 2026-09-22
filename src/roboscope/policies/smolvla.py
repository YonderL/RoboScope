"""LeRobot SmolVLA with strict pretrained loading and native preprocessing/loss."""

from pathlib import Path

import torch
from torch import nn

from roboscope.runtime.common import CAMERAS

IMAGE_KEYS = ("observation.images.camera1", "observation.images.camera2")
ASSET_KEYS = ("pretrained_path", "pretrained_revision", "vlm_path", "vlm_revision")


def resolve_assets(cfg):
    from huggingface_hub import HfApi, snapshot_download

    from roboscope.policies.pi0 import _local_revision

    cache = str(Path(cfg.get("hf_cache_dir", ".cache/smolvla/hub")).expanduser().resolve())
    result = {}
    for kind, patterns in (
        ("pretrained", ["config.json", "model.safetensors"]),
        ("vlm", ["*.json", "*.txt", "*.model", "*.jinja"]),
    ):
        source = cfg[f"{kind}_path"]
        directory = Path(source).expanduser()
        if directory.is_dir():
            revision = _local_revision(directory, [p for p in directory.rglob("*") if p.is_file()])
        else:
            revision = HfApi().model_info(source, revision=cfg[f"{kind}_revision"]).sha
            directory = Path(
                snapshot_download(source, revision=revision, cache_dir=cache, allow_patterns=patterns)
            )
        for name in ["config.json", "model.safetensors"] if kind == "pretrained" else ["config.json"]:
            if not (directory / name).is_file():
                raise FileNotFoundError(directory / name)
        result[f"{kind}_path"] = str(directory.resolve())
        result[f"{kind}_revision"] = revision
    return result


def load_backend(cfg, initialize_pretrained=True):
    from lerobot.configs import FeatureType, PolicyFeature
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from safetensors.torch import load_model

    # Keep architecture, freeze settings, optimizer and scheduler from the
    # published base config. Only adapt robot features and local asset paths.
    config = SmolVLAConfig.from_pretrained(cfg["pretrained_path"])
    if (
        config.chunk_size != 50
        or config.n_action_steps != 50
        or config.max_action_dim != 32
        or config.n_obs_steps != 1
    ):
        raise ValueError("Expected the official SmolVLA base architecture (chunk=50, padded action=32)")
    config.device = "cpu"
    config.vlm_model_name = cfg["vlm_path"]
    pretrained_vlm_dtype = config.load_vlm_weights
    config.load_vlm_weights = False  # The complete VLM weights come from smolvla_base below.
    config.input_features = {key: PolicyFeature(FeatureType.VISUAL, (3, 128, 128)) for key in IMAGE_KEYS}
    config.input_features["observation.state"] = PolicyFeature(FeatureType.STATE, (9,))
    config.output_features = {"action": PolicyFeature(FeatureType.ACTION, (7,))}
    config.empty_cameras = 0
    if config.adapt_to_pi_aloha or config.use_delta_joint_actions_aloha:
        raise ValueError("LIBERO requires raw OSC actions, without Aloha transforms")
    policy = SmolVLAPolicy(config)
    if pretrained_vlm_dtype:
        # Native load_vlm_weights=True creates the frozen VLM in BF16, while
        # the expert/projections remain FP32. Preserve that mixed parameter
        # precision without redundantly downloading the standalone VLM weights.
        policy.model.vlm_with_expert.vlm.to(dtype=torch.bfloat16)
    if initialize_pretrained:
        # Unlike permissive Hub loaders, missing/incompatible weights abort.
        # load_model supports tied VLM embeddings omitted by safetensors.
        load_model(policy, str(Path(cfg["pretrained_path"]) / "model.safetensors"), strict=True)
    return policy


class SmolVLAPolicy(nn.Module):
    def __init__(self, cfg, manifest, initialize_pretrained=True):
        super().__init__()
        from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

        self.policy = load_backend(cfg, initialize_pretrained)
        self.languages = {task["id"]: task["language"] for task in manifest["tasks"]}
        stats = {
            target: {
                name: torch.tensor(manifest[f"{source}_{name}"], dtype=torch.float32)
                for name in ("mean", "std")
            }
            for source, target in (("state", "observation.state"), ("action", "action"))
        }
        self.preprocessor, self.postprocessor = make_smolvla_pre_post_processors(self.policy.config, stats)

    def prepare_batch(self, batch, include_action=False):
        # Set the native processor device after .to()/.cuda(); these processors
        # are not nn.Modules and do not follow module device transfers.
        device = next(self.policy.parameters()).device
        from lerobot.processor import DeviceProcessorStep

        for step in self.preprocessor.steps:
            if isinstance(step, DeviceProcessorStep) and step.tensor_device != device:
                step.device = str(device)
                step.__post_init__()
        tasks = batch.get("task")
        if tasks is None:
            tasks = [self.languages[int(i)] for i in batch["task_id"].tolist()]
        result = {"observation.state": batch["state"], "task": tasks}
        for source, target in zip(CAMERAS, IMAGE_KEYS, strict=True):
            result[target] = batch[source].float() / 255.0
        if include_action:
            result.update(action=batch["action"], action_is_pad=batch["action_is_pad"])
        return self.preprocessor(result)

    def forward(self, batch):
        loss, _ = self.policy(self.prepare_batch(batch, include_action=True))
        return loss

    @torch.no_grad()
    def predict(self, batch, noise=None):
        prepared = self.prepare_batch(batch)
        actions = self.policy.predict_action_chunk(prepared, noise=noise)
        return self.postprocessor(actions)
