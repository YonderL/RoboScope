"""Strict pretrained Pi-0 loading and small, native PyTorch LoRA checkpoints.

LeRobot 0.6.1 supplies the architecture and flow matching implementation. Its
``from_pretrained`` catches loading errors, so this adapter deliberately uses
explicit safetensors loading: a missing/incompatible base must stop the run.
"""

import hashlib
import math
import os
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from roboscope.runtime.common import CAMERAS

PROJECTIONS = ("state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out")
ATTENTION_TARGET = re.compile(
    r"model\.paligemma_with_expert\."
    r"(?:paligemma\.model\.language_model|gemma_expert\.model)"
    r"\.layers\.\d+\.self_attn\.(?:q|k|v|o)_proj$"
)
IMAGE_KEYS = ("observation.images.base_0_rgb", "observation.images.left_wrist_0_rgb")
PUBLIC_TOKENIZER = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"


class OpenPiTokenizer:
    """Official Pi-0 SentencePiece format: BOS + instruction + separate newline."""

    def __init__(self, path):
        import sentencepiece

        self.processor = sentencepiece.SentencePieceProcessor(model_file=str(path))
        if self.processor.bos_id() != 2 or self.processor.pad_id() != 0:
            raise ValueError("Tokenizer does not have the expected PaliGemma special-token IDs")

    def __call__(self, prompt, max_length=48, **kwargs):
        prompt = prompt.strip().replace("_", " ").replace("\n", " ")
        ids = self.processor.encode(prompt, add_bos=True) + self.processor.encode("\n")
        ids = ids[:max_length]
        mask = [1] * len(ids) + [0] * (max_length - len(ids))
        ids += [0] * (max_length - len(ids))
        return {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([mask])}


def load_tokenizer(directory):
    directory = Path(directory)
    public = directory / "paligemma_tokenizer.model"
    if public.is_file():
        return OpenPiTokenizer(public)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)


def _local_revision(directory, files):
    """Hub snapshots already have immutable names; fingerprint standalone assets."""
    if directory.parent.name == "snapshots" and re.fullmatch(r"[0-9a-f]{40}", directory.name):
        return directory.name
    digest = hashlib.sha256()
    for path in sorted(files):
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            while block := handle.read(8 * 1024 * 1024):
                digest.update(block)
    return "local-sha256:" + digest.hexdigest()


def cached_pretrained_directory(cache_dir, source, revision):
    """Only a complete, explicitly pinned snapshot can bypass the Hub API."""
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        return None
    directory = Path(cache_dir) / ("models--" + source.replace("/", "--")) / "snapshots" / revision
    if all((directory / name).is_file() for name in ("config.json", "model.safetensors")):
        return directory
    return None


def resolve_assets(cfg):
    """Download once before torchrun and resolve both Hub refs to immutable SHAs.

    HF_ENDPOINT / HF_TOKEN are handled by huggingface_hub itself. Tokens are never
    written into the recipe or returned metadata. Local asset directories work
    without network or account access.
    """
    from huggingface_hub import HfApi, snapshot_download

    resolved = {}
    cache_dir = str(Path(cfg.get("hf_cache_dir", ".cache/pi0/hub")).expanduser().resolve())
    for kind, patterns in (
        ("tokenizer", ["tokenizer*", "special_tokens_map.json", "added_tokens.json", "config.json"]),
        ("pretrained", ["config.json", "model.safetensors"]),
    ):
        source = cfg[f"{kind}_path"]
        directory = Path(source).expanduser()
        cached = (
            cached_pretrained_directory(cache_dir, source, cfg.get("pretrained_revision", "main"))
            if kind == "pretrained" and not directory.is_dir()
            else None
        )
        if cached is not None:
            resolved["pretrained_path"] = str(cached.resolve())
            resolved["pretrained_revision"] = cfg["pretrained_revision"]
            continue
        if kind == "tokenizer" and source == PUBLIC_TOKENIZER:
            import requests

            directory = Path(cache_dir).parent / "openpi_tokenizer"
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / "paligemma_tokenizer.model"
            if not target.exists():
                temporary = target.with_suffix(".tmp")
                try:
                    with requests.get(PUBLIC_TOKENIZER, stream=True, timeout=(15, 60)) as response:
                        response.raise_for_status()
                        with temporary.open("wb") as handle:
                            for block in response.iter_content(1024 * 1024):
                                handle.write(block)
                            handle.flush()
                            os.fsync(handle.fileno())
                    OpenPiTokenizer(temporary)
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
            revision = _local_revision(directory, [target])
        elif directory.is_dir():
            directory = directory.resolve()
            files = [p for p in directory.iterdir() if p.is_file() and not p.name.startswith(".")]
            revision = _local_revision(directory, files)
        else:
            revision = cfg.get(f"{kind}_revision", "main")
            try:
                if not cfg.get("local_files_only", False) and not re.fullmatch(r"[0-9a-f]{40}", revision):
                    revision = HfApi().model_info(source, revision=revision).sha
                directory = Path(
                    snapshot_download(
                        repo_id=source,
                        revision=revision,
                        cache_dir=cache_dir,
                        allow_patterns=patterns,
                        max_workers=1,
                        local_files_only=cfg.get("local_files_only", False),
                    )
                ).resolve()
                revision = directory.name
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot prepare {kind} assets from {source!r}. Check Hub connectivity/access "
                    "or set PI0_BASE_PATH / PI0_TOKENIZER_PATH to existing local asset directories. "
                    "No randomly initialized Pi-0 fallback is allowed."
                ) from exc
        if kind == "pretrained":
            for filename in ("config.json", "model.safetensors"):
                if not (directory / filename).is_file():
                    raise FileNotFoundError(f"Missing pretrained asset: {directory / filename}")
        else:
            # Check permissions/files/tokenizer compatibility before downloading 14 GB of weights.
            load_tokenizer(directory)
        resolved[f"{kind}_path"] = str(directory)
        resolved[f"{kind}_revision"] = revision
    return resolved


class LoRALinear(nn.Module):
    """Frozen dense base with FP32 rank-r adapters, compatible with Pi0 weight inspection."""

    def __init__(self, base, rank, alpha, dropout=0.0):
        super().__init__()
        if rank < 1 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid LoRA rank, alpha or dropout")
        self.base = base.requires_grad_(False)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, inputs):
        output = self.base(inputs)
        update = F.linear(F.linear(self.dropout(inputs.float()), self.lora_A), self.lora_B)
        return output + update.to(output.dtype) * self.scaling


def inject_lora(policy, cfg):
    """Only transformer attention Q/K/V/O; the vision encoder remains frozen."""
    if cfg.get("gradient_checkpointing", True) and cfg.get("lora_dropout", 0.0):
        raise ValueError("LeRobot checkpoints do not preserve RNG; use lora_dropout=0 with checkpointing")
    policy.requires_grad_(False)
    targets = [
        name
        for name, module in policy.named_modules()
        if isinstance(module, nn.Linear) and ATTENTION_TARGET.fullmatch(name)
    ]
    if not targets:
        raise ValueError("No Pi-0 attention projections matched; incompatible LeRobot architecture")
    for name in targets:
        parent_name, leaf = name.rsplit(".", 1)
        parent = policy.get_submodule(parent_name)
        setattr(
            parent,
            leaf,
            LoRALinear(
                getattr(parent, leaf),
                cfg.get("lora_rank", 16),
                cfg.get("lora_alpha", 16),
                cfg.get("lora_dropout", 0.0),
            ),
        )
    for name in PROJECTIONS:
        policy.model.get_submodule(name).float().requires_grad_(True)
    return targets


def _load_backend(cfg):
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from safetensors.torch import load_file

    root = Path(cfg["pretrained_path"])
    if not root.is_dir():
        raise ValueError("Resolve pretrained assets with resolve_assets(cfg) before creating Pi0Policy")
    model_cfg = PI0Config.from_pretrained(root)
    expected = {
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "max_state_dim": 32,
        "max_action_dim": 32,
        "chunk_size": 50,
    }
    for name, value in expected.items():
        if getattr(model_cfg, name) != value:
            raise ValueError(f"Incompatible pretrained Pi-0 {name}: expected {value}")
    if cfg.get("chunk_size", 50) != 50 or cfg.get("max_action_dim", 32) != 32:
        raise ValueError("Pi-0 base requires chunk_size=50 and max_action_dim=32")
    model_cfg.device = "cpu"
    model_cfg.dtype = cfg.get("dtype", "bfloat16")
    model_cfg.gradient_checkpointing = cfg.get("gradient_checkpointing", True)
    model_cfg.freeze_vision_encoder = True
    model_cfg.train_expert_only = False
    model_cfg.compile_model = False
    # LIBERO actions already encode OSC deltas; never subtract state again.
    model_cfg.use_relative_actions = False
    model_cfg.n_action_steps = cfg.get("action_horizon", 8)
    model_cfg.num_inference_steps = cfg.get("num_inference_steps", 10)
    model_cfg.tokenizer_max_length = cfg.get("tokenizer_max_length", 48)
    model_cfg.input_features = {key: PolicyFeature(FeatureType.VISUAL, (3, 224, 224)) for key in IMAGE_KEYS}
    model_cfg.input_features["observation.state"] = PolicyFeature(FeatureType.STATE, (8,))
    model_cfg.output_features = {"action": PolicyFeature(FeatureType.ACTION, (7,))}
    policy = PI0Policy(model_cfg)
    raw = load_file(str(root / "model.safetensors"), device="cpu")
    fixed = policy._fix_pytorch_state_dict_keys(raw, model_cfg)
    weights = {key if key.startswith("model.") else f"model.{key}": value for key, value in fixed.items()}
    # Do not catch: missing, unexpected and mismatched weights must abort training.
    policy.load_state_dict(weights, strict=True)
    return policy


class Pi0Policy(nn.Module):
    """Project batch -> normalized, language-conditioned Pi-0 -> raw OSC action chunk."""

    def __init__(self, cfg, manifest):
        super().__init__()
        self.cfg = cfg
        self.provenance = {
            "backend": "lerobot.pi0",
            "adapter": "native_torch_lora",
            **{
                key: cfg.get(key)
                for key in ("pretrained_path", "pretrained_revision", "tokenizer_path", "tokenizer_revision")
            },
            "state_dim": 8,
            "action_dim": 7,
            "padded_dim": 32,
            "chunk_size": 50,
            "normalization": "training_split_mean_std",
        }
        self.policy = _load_backend(cfg)
        self.lora_targets = inject_lora(self.policy, cfg)
        self.tokenizer = load_tokenizer(cfg["tokenizer_path"])
        self.tokenizer.padding_side = "right"
        self._tokens = {}
        for name, size in (("state_mean", 8), ("state_std", 8), ("action_mean", 7), ("action_std", 7)):
            value = torch.tensor(manifest[f"pi0_{name}"], dtype=torch.float32)
            if value.shape != (size,) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid Pi-0 training statistics: {name}")
            if name.endswith("std") and (value <= 0).any():
                raise ValueError(f"Pi-0 {name} must be positive")
            self.register_buffer(name, value)

    def prepare_batch(self, batch, include_action=False):
        state = batch["state"]
        if state.ndim != 2 or state.shape[-1] != 8:
            raise ValueError("Pi-0 expects [B,8] end-effector xyz/axis-angle/gripper state")
        tasks = batch["task"]
        if len(tasks) != len(state) or not all(isinstance(t, str) and t.strip() for t in tasks):
            raise ValueError("Pi-0 requires one nonempty natural-language task per sample")
        for task in set(tasks):
            if task not in self._tokens:
                prompt = task if task.endswith("\n") else task + "\n"
                tokens = self.tokenizer(
                    prompt,
                    max_length=self.cfg.get("tokenizer_max_length", 48),
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                self._tokens[task] = (tokens["input_ids"][0], tokens["attention_mask"][0].bool())
        result = {
            "observation.state": (state.float() - self.state_mean) / self.state_std,
            "observation.language.tokens": torch.stack([self._tokens[t][0] for t in tasks]).to(state.device),
            "observation.language.attention_mask": torch.stack([self._tokens[t][1] for t in tasks]).to(
                state.device
            ),
        }
        for source, target in zip(CAMERAS, IMAGE_KEYS, strict=True):
            images = batch[source]
            if images.ndim != 4 or images.shape[1] != 3 or images.dtype != torch.uint8:
                raise ValueError(f"Pi-0 camera {source} must be uint8 [B,3,H,W] RGB")
            # FrameDataset and rollout already use OpenCV image orientation;
            # these are RGB channels, so do not perform a BGR channel swap.
            result[target] = images.float() / 255.0
        if include_action:
            actions = batch["action"]
            if actions.shape != (len(state), 50, 7):
                raise ValueError("Pi-0 expects a [B,50,7] raw OSC action chunk")
            result["action"] = (actions.float() - self.action_mean) / self.action_std
        return result

    def forward(self, batch):
        prepared = self.prepare_batch(batch, include_action=True)
        images, masks = self.policy._preprocess_images(prepared)
        state = self.policy.prepare_state(prepared)
        actions = self.policy.prepare_action(prepared)
        noise = self.policy.model.sample_noise(actions.shape, actions.device)
        time = self.policy.model.sample_time(actions.shape[0], actions.device)
        losses = self.policy.model(
            images,
            masks,
            prepared["observation.language.tokens"],
            prepared["observation.language.attention_mask"],
            state,
            actions,
            noise,
            time,
        )[..., :7]
        if self.cfg.get("mask_padding_loss", False):
            mask = (~batch["action_is_pad"]).unsqueeze(-1)
            return (losses.float() * mask).sum() / (mask.sum() * 7).clamp_min(1)
        return losses.float().mean()

    @torch.no_grad()
    def predict(self, batch, noise=None):
        prepared = self.prepare_batch(batch)
        if noise is not None and noise.shape != (len(batch["state"]), 50, 32):
            raise ValueError("Pi-0 inference noise must have the full padded shape [B,50,32]")
        actions = self.policy.predict_action_chunk(prepared, noise=noise)
        return actions.float() * self.action_std + self.action_mean

    def trainable_state_dict(self):
        """Portable adapter + robot projection weights; immutable base is saved by reference."""
        return {
            name: param.detach().cpu().clone()
            for name, param in self.named_parameters()
            if param.requires_grad
        }

    def load_trainable_state_dict(self, state):
        targets = {name: param for name, param in self.named_parameters() if param.requires_grad}
        if state.keys() != targets.keys():
            raise ValueError(
                f"Adapter keys differ: missing={targets.keys() - state.keys()}, "
                f"unexpected={state.keys() - targets.keys()}"
            )
        for name, param in targets.items():
            if state[name].shape != param.shape:
                raise ValueError(f"Adapter shape differs for {name}")
        with torch.no_grad():
            for name, param in targets.items():
                param.copy_(state[name])

    def parameter_counts(self):
        return {
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total": sum(p.numel() for p in self.parameters()),
            "lora_modules": len(self.lora_targets),
        }
