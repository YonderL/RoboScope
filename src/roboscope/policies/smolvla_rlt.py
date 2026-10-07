"""Frozen SmolVLA -> final image-position tokens -> RL token -> direct action actor."""

import json
from pathlib import Path

import torch
from torch import nn

from roboscope.policies.smolvla import SmolVLAPolicy
from roboscope.rl.networks import Actor
from roboscope.rl.token import RLToken
from roboscope.runtime.common import digest


def image_features_from_prefix(features, valid, attention, language_length):
    """Select native image positions, excluding language/state and trailing padding."""
    # 原生布局：[image tokens][language（包含 padding）][state][可选 padding]。
    # 第一个 attention=1 标记 state 起点；语言长度包含被 mask 的 padding 位置。
    state_start = attention[0].to(torch.int64).argmax()
    image_end = state_start - language_length
    if image_end <= 0:
        raise ValueError("No image tokens found in the native SmolVLA prefix")
    positions = torch.arange(features.shape[1], device=features.device) < image_end
    # 不按 valid.sum() 截断：缺失相机仍占据原位置，但由 image_valid 屏蔽。
    return features[:, positions].detach().float(), valid[:, positions].bool()


def load_sft(cfg, manifest, device):
    if cfg.get("sft_backend") == "hf_native":
        from roboscope.data.smolvla_hf import verify_contract
        from roboscope.policies.smolvla_hf import HFSmolVLAPolicy

        verify_contract(cfg, manifest)
        return HFSmolVLAPolicy(cfg["sft_checkpoint"], manifest, device).eval().requires_grad_(False)

    from roboscope.runtime.training import manifest_digest

    path = cfg["sft_checkpoint"]
    if digest(path) != cfg["sft_sha256"]:
        raise ValueError("The frozen SFT checkpoint changed")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    if saved.get("format") != "roboscope.smolvla.v1":
        raise ValueError("RLT requires a trained RoboScope SmolVLA SFT checkpoint")
    # Evaluation may replace eval_initial_state_ids, but training statistics,
    # language instructions, task identities and data split must stay unchanged.
    source_manifest = json.loads((Path(cfg["sft_source"]) / "manifest.json").read_text())
    if saved.get("manifest_sha256") != manifest_digest(source_manifest):
        raise ValueError("Frozen SFT manifest changed")

    def without_schedule(value):
        return {
            **value,
            "tasks": [
                {k: v for k, v in task.items() if k != "eval_initial_state_ids"} for task in value["tasks"]
            ],
        }

    if without_schedule(manifest) != without_schedule(source_manifest):
        raise ValueError("RLT and SFT manifest differ beyond the evaluation schedule")
    model = SmolVLAPolicy(saved["config"], manifest, initialize_pretrained=False)
    model.load_state_dict(saved["model"], strict=True)
    return model.to(device).eval().requires_grad_(False)


@torch.no_grad()
def prefix_prefill(base, batch):
    """Return image features/mask, full prefix mask and unmodified native KV cache.

    RLT Fig. 2 / footnote 1: reconstruct image tokens only. Language and state
    still enter the frozen VLA normally so its reference actions are unchanged.
    """
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    prepared = base.prepare_batch(batch)
    policy, model = base.policy, base.policy.model
    if model.add_image_special_tokens:
        raise ValueError("RLT image selection requires SmolVLA base's add_image_special_tokens=False")
    images, image_masks = policy.prepare_images(prepared)
    state = policy.prepare_state(prepared)
    embeddings, valid, attention = model.embed_prefix(
        images,
        image_masks,
        prepared["observation.language.tokens"],
        prepared["observation.language.attention_mask"],
        state=state,
    )
    # 提取的是 VLM 文本/多模态 Transformer 主干处理后的 prefix 特征：
    # 图像、语言、状态 embeddings -> 已保留的全部 VLM blocks -> text_model.norm。
    # 官方 smolvla_base 的 num_vlm_layers=16，即第 16 个 block（索引 15）之后，
    # 再经过最终 RMSNorm；不是原始完整 SmolVLM 的最后一层，也不是视觉编码器输出。
    # 实际深度跟随载入的 SFT 架构：model.vlm_with_expert.num_vlm_layers，未硬编码 16。
    outputs, cache = model.vlm_with_expert.forward(
        attention_mask=make_att_2d_masks(valid, attention),
        position_ids=valid.cumsum(-1) - 1,
        past_key_values=None,
        inputs_embeds=[embeddings, None],
        use_cache=True,
    )
    # outputs[0] 的 0 是 VLM 分支编号，不是第 0 层；outputs[1] 是 action expert 分支。
    image_features, image_valid = image_features_from_prefix(
        outputs[0], valid, attention, prepared["observation.language.tokens"].shape[1]
    )
    # 只把图像位置送入 RLToken encoder 和重建目标；语言/state 位置均排除。
    # 图像输出仍可包含 VLM 内部注意力融合的语言信息，不等于移除 VLA 的语言输入。
    # flow action expert 需要完整 prefix mask/cache，不能随图像特征一起裁剪。
    return image_features, image_valid, valid.bool(), cache


@torch.no_grad()
def vlm_features(base, batch):
    features, valid, _, _ = prefix_prefill(base, batch)
    return features, valid


class RLTPolicy(nn.Module):
    def __init__(self, base, token, actor, cfg, manifest):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.token, self.actor, self.cfg = token, actor, cfg
        self.register_buffer("state_mean", torch.tensor(manifest["state_mean"], dtype=torch.float32))
        self.register_buffer("state_std", torch.tensor(manifest["state_std"], dtype=torch.float32))
        self.state_eps = manifest.get("state_eps", 0.0)

    @torch.no_grad()
    def describe(self, batch, noise=None):
        """One VLA call yields both its full reference action and prefix tokens."""
        from lerobot.policies.common.flow_matching import euler_integrate

        features, image_valid, prefix_valid, cache = prefix_prefill(self.base, batch)
        device = self.state_mean.device
        model = self.base.policy.model
        if noise is None:
            noise = model.sample_noise((len(features), 50, 32), device)
        # Same Euler solver and denoise_step as native sample_actions; reuse
        # its prefix cache for both the RL representation and the reference.
        actions = euler_integrate(
            lambda x, t: model.denoise_step(prefix_valid, cache, x, t),
            noise,
            model.config.num_steps,
        )[..., :7]
        reference = self.base.postprocessor(actions)
        # postprocessor 将标准化动作还原为环境控制量；actor/critic 都使用此动作尺度。
        # SmolVLA 仍生成 H=50 步，最后仅截取前 C=10 步作为 actor 的参考。
        # The small trainable modules stay FP32, independently of VLA autocast.
        with torch.autocast(device.type, enabled=False):
            token = self.token.encode(features, image_valid)
            state = (batch["state"].to(device).float() - self.state_mean) / (self.state_std + self.state_eps)
            # 本体状态不参与 RL token 重建；在此才与 image-only RL token 拼接，
            # 供 actor 和 critic 使用：HF 为 8D EEF，旧版为 9D joints。
            remaining = batch.get(
                "remaining_steps", torch.full((len(state),), self.cfg["rollout_horizon"], device=device)
            )
            remaining = remaining.to(device).float().reshape(-1, 1) / self.cfg["rollout_horizon"]
            # 相同画面在剩 500 步与剩 1 步时价值不同；剩余时间使截止条件进入状态。
            rl_state = torch.cat([token, state, remaining], dim=-1)
        return rl_state, reference[:, : self.cfg["action_horizon"]].to(device).float().clamp(-1, 1)

    @torch.no_grad()
    def predict(self, batch, noise=None):
        if self.cfg.get("evaluation_policy") == "sft_reference":
            return self.base.predict(batch, noise=noise)[:, : self.cfg["action_horizon"]]
        state, reference = self.describe(batch, noise)
        with torch.autocast(state.device.type, enabled=False):
            return self.actor(state, reference)  # Gaussian mean for evaluation


def build_policy(base, cfg, manifest):
    if cfg.get("token_features") != "vlm_image_tokens":
        raise ValueError("RLT requires image-only token features; retrain old full-prefix token/replay")
    width = base.policy.model.vlm_with_expert.config.text_config.hidden_size
    # token_layers 是 encoder、decoder 各自的深度：默认各 2 层，共 4 个独立 block。
    # 生产路径要求 token_dim 等于 VLM hidden_size（SmolVLM2-500M 为 960），
    # 直接使用图像 embedding，禁止静默降维到更窄的 readout。
    if cfg["token_dim"] != width:
        raise ValueError(
            f"token_dim={cfg['token_dim']} must equal VLM hidden_size={width}; "
            "do not down-project image embeddings before the RL token"
        )
    token = RLToken(width, cfg["token_dim"], cfg["token_layers"], cfg["token_heads"])
    state_dim = len(manifest["state_mean"])
    if cfg.get("state_dim", 9) != state_dim:
        raise ValueError("RLT state dimension differs from the SFT normalization statistics")
    actor = Actor(
        cfg["token_dim"] + state_dim + 1, cfg["action_horizon"], cfg["hidden_dims"], cfg["actor_std"]
    )
    return RLTPolicy(base, token, actor, cfg, manifest)


def load_policy(checkpoint, manifest, evaluation_cfg, device):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if saved.get("format") != "roboscope.smolvla_rlt.v1":
        raise ValueError("Not a SmolVLA RLT checkpoint")
    if saved["config"].get("sft_backend") == "hf_native":
        from roboscope.runtime.training import manifest_digest

        original = json.loads((Path(saved["config"]["output_root"]) / "manifest.json").read_text())
        if saved["manifest_sha256"] != manifest_digest(original):
            raise ValueError("HF RLT training manifest changed")

        def identity(value):
            return {
                **value,
                "tasks": [
                    {k: v for k, v in t.items() if k != "eval_initial_state_ids"} for t in value["tasks"]
                ],
            }

        if identity(original) != identity(manifest):
            raise ValueError("HF RLT evaluation manifest differs beyond the initial-state schedule")
    cfg = {**saved["config"], "evaluation_policy": evaluation_cfg.get("evaluation_policy", "rlt")}
    base = load_sft(cfg, manifest, device)
    policy = build_policy(base, cfg, manifest).to(device)
    policy.token.load_state_dict(saved["token"], strict=True)
    policy.actor.load_state_dict(saved["actor"], strict=True)
    return policy.eval().requires_grad_(False), saved["env_steps"]
