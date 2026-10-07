"""Strict native SmolVLA checkpoint loading with its saved processor statistics."""

import json
from pathlib import Path

import torch
from torch import nn

from roboscope.policies.smolvla import SmolVLAPolicy as LegacyPolicy
from roboscope.runtime.common import CAMERAS, digest

IMAGE_KEYS = ("observation.images.image", "observation.images.image2")


def checkpoint_files(path):
    path = Path(path)
    names = {"config.json", "model.safetensors", "policy_preprocessor.json", "policy_postprocessor.json"}
    for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
        for step in json.loads((path / name).read_text())["steps"]:
            if step.get("state_file"):
                names.add(step["state_file"])
    return {name: digest(path / name) for name in sorted(names)}


def validate_config(config):
    shapes = {k: v["shape"] for k, v in config["input_features"].items()}
    if shapes != {**dict.fromkeys(IMAGE_KEYS, [3, 256, 256]), "observation.state": [8]}:
        raise ValueError("Expected native HF Spatial inputs: two 256px cameras and 8D EEF state")
    if (config["chunk_size"], config["max_action_dim"], config["n_obs_steps"]) != (50, 32, 1):
        raise ValueError("Expected native 50x32 SmolVLA action proposal")
    if config["output_features"]["action"]["shape"] != [7]:
        raise ValueError("Expected raw 7D OSC actions")
    if config["adapt_to_pi_aloha"] or config["use_delta_joint_actions_aloha"]:
        raise ValueError("HF LIBERO must not use Aloha action transforms")


class HFSmolVLAPolicy(nn.Module):
    def __init__(self, checkpoint, manifest, device="cpu"):
        super().__init__()
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        from safetensors.torch import load_model

        checkpoint = Path(checkpoint)
        validate_config(json.loads((checkpoint / "config.json").read_text()))
        config = SmolVLAConfig.from_pretrained(checkpoint)
        config.device, config.compile_model = str(device), False
        mixed_vlm = config.load_vlm_weights
        config.load_vlm_weights = False  # Complete weights come strictly from this SFT checkpoint.
        self.policy = SmolVLAPolicy(config)
        if mixed_vlm:
            self.policy.model.vlm_with_expert.vlm.to(dtype=torch.bfloat16)
        load_model(self.policy, str(checkpoint / "model.safetensors"), strict=True)
        self.policy.to(device)
        # Match LeRobot's CPU uint8 / 255 rounding exactly on CUDA as well.
        # A reciprocal multiply on CUDA can differ by one FP32 ULP before BF16 inference.
        self.register_buffer(
            "pixel_values", (torch.arange(256, dtype=torch.float32) / 255).to(device), persistent=False
        )
        self.languages = {task["id"]: task["language"] for task in manifest["tasks"]}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            config,
            pretrained_path=str(checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )

    def prepare_batch(self, batch):
        from lerobot.processor import DeviceProcessorStep

        device = next(self.policy.parameters()).device
        for step in self.preprocessor.steps:
            if isinstance(step, DeviceProcessorStep) and step.tensor_device != device:
                step.device = str(device)
                step.__post_init__()
        if batch["state"].shape[-1] != 8:
            raise ValueError("HF SmolVLA requires an 8D end-effector state")
        tasks = batch.get("task")
        if tasks is None:
            tasks = [self.languages[int(i)] for i in batch["task_id"].tolist()]
        native = {"observation.state": batch["state"], "task": tasks}
        for source, target in zip(CAMERAS, IMAGE_KEYS, strict=True):
            if batch[source].shape[-3:] != (3, 256, 256):
                raise ValueError("HF SmolVLA requires 256x256 RGB inputs")
            if batch[source].dtype != torch.uint8:
                raise ValueError("HF adapter expects raw uint8 pixels in dataset orientation")
            native[target] = self.pixel_values[batch[source].to(device=device, dtype=torch.long)]
        return self.preprocessor(native)

    predict = LegacyPolicy.predict
