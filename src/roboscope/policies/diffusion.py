"""CNN + FiLM Conditional 1D U-Net, epsilon prediction, DDIM and task-ID conditioning."""

import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionConditionalUnet1d, DiffusionRgbEncoder
from torch import nn

from roboscope.runtime.common import CAMERAS


class TaskDiffusionPolicy(nn.Module):
    def __init__(self, cfg, manifest):
        super().__init__()
        self.cfg = cfg
        dc = DiffusionConfig(
            n_obs_steps=2,
            horizon=16,
            n_action_steps=8,
            input_features={
                "observation.state": PolicyFeature(FeatureType.STATE, (9,)),
                "observation.images.rgb": PolicyFeature(FeatureType.VISUAL, (3, 128, 128)),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
            down_dims=tuple(cfg["down_dims"]),
            kernel_size=cfg["kernel_size"],
            diffusion_step_embed_dim=cfg["diffusion_step_embed_dim"],
            spatial_softmax_num_keypoints=cfg["spatial_keypoints"],
            crop_shape=tuple(cfg["crop_shape"]),
            resize_shape=None,
            use_group_norm=True,
            pretrained_backbone_weights=None,
        )
        # 原版 DP 的多相机设计：每路图像拥有独立、从零训练的 ResNet。
        # GN 在单个样本内按通道组归一化，没有 BN 的 running statistics，适合 EMA。
        # 每个组 16 个通道，与原版 MultiImageObsEncoder 的替换规则一致。
        self.encoders = nn.ModuleDict({camera: DiffusionRgbEncoder(dc) for camera in CAMERAS})
        # 禁用 LeRobot 的 batch 共用随机 crop，改为下面逐图像采样位置。
        for encoder in self.encoders.values():
            encoder.do_crop = False
        self.task_embedding = nn.Embedding(len(manifest["tasks"]), cfg["task_embedding_dim"])
        cond_dim = (
            2 * (sum(encoder.feature_dim for encoder in self.encoders.values()) + 9)
            + cfg["task_embedding_dim"]
        )
        self.unet = DiffusionConditionalUnet1d(dc, cond_dim)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=cfg["diffusion_steps"],
            beta_schedule=cfg["beta_schedule"],
            prediction_type="epsilon",
            clip_sample=cfg["clip_sample"],
            timestep_spacing="leading",
            set_alpha_to_one=True,
        )
        for key in ("state_mean", "state_std", "action_mean", "action_std"):
            self.register_buffer(key, torch.tensor(manifest[key], dtype=torch.float32))
        low = torch.tensor(manifest["action_min"], dtype=torch.float32)
        high = torch.tensor(manifest["action_max"], dtype=torch.float32)
        # 常量维使用单位 scale 并平移到零，避免除零；与原版 limits 处理一致。
        span = high - low
        scale = torch.where(span < 1e-4, torch.ones_like(span), 2 / span.clamp_min(1e-4))
        offset = torch.where(span < 1e-4, -low, -1 - low * scale)
        self.register_buffer("action_scale", scale)
        self.register_buffer("action_offset", offset)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def crop_images(self, images):
        """训练每幅图独立随机裁剪；验证/推理固定中心裁剪，无随机抖动。"""
        n, _, h, w = images.shape
        ch, cw = self.cfg["crop_shape"]
        if self.training:
            top = torch.randint(h - ch + 1, (n,), device=images.device)
            left = torch.randint(w - cw + 1, (n,), device=images.device)
        else:
            top = torch.full((n,), (h - ch) // 2, device=images.device, dtype=torch.long)
            left = torch.full((n,), (w - cw) // 2, device=images.device, dtype=torch.long)
        rows = top[:, None, None] + torch.arange(ch, device=images.device)[None, :, None]
        cols = left[:, None, None] + torch.arange(cw, device=images.device)[None, None, :]
        return (
            images.permute(0, 2, 3, 1)[torch.arange(n, device=images.device)[:, None, None], rows, cols]
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    def condition(self, batch):
        features = []
        for camera in CAMERAS:
            images = batch[camera]
            b, t = images.shape[:2]
            images = (self.crop_images(images.flatten(0, 1)).float() / 255 - self.image_mean) / self.image_std
            features.append(self.encoders[camera](images).reshape(b, t, -1))
        features.append((batch["state"] - self.state_mean) / self.state_std)
        return torch.cat(
            [torch.cat(features, dim=-1).flatten(1), self.task_embedding(batch["task_id"].long())], -1
        )

    def forward(self, batch):
        """训练抽一个随机噪声等级；100 是噪声调度长度，不是每个 batch 前向 100 次。"""
        clean = batch["action"] * self.action_scale + self.action_offset
        noise = torch.randn_like(clean)
        timesteps = torch.randint(0, self.cfg["diffusion_steps"], (len(clean),), device=clean.device)
        noisy = self.scheduler.add_noise(clean, noise, timesteps)
        predicted = self.unet(noisy, timesteps, self.condition(batch))
        # 原版对复制的边界动作也监督；保留显式开关以记录实现差异。
        if not self.cfg["mask_padding_loss"]:
            return F.mse_loss(predicted.float(), noise)
        mask = (~batch["action_is_pad"]).unsqueeze(-1)
        return ((predicted.float() - noise).square() * mask).sum() / (mask.sum() * 7).clamp_min(1)

    @torch.no_grad()
    def predict(self, batch, ddim_steps=10, noise=None):
        """标准化空间去噪；窗口下标 1 对应当前时刻，返回未来 15 步。

        同一个 checkpoint 只改 scheduler 步数；Ta 在执行队列中改变。
        预先提供 noise 可以让每个 episode 的采样独立于并行槽位顺序。
        """
        cond = self.condition(batch)
        sample = noise if noise is not None else torch.randn(len(cond), 16, 7, device=cond.device)
        self.scheduler.set_timesteps(ddim_steps, device=sample.device)
        for t in self.scheduler.timesteps:
            prediction = self.unet(sample, t.expand(len(sample)), cond)
            sample = self.scheduler.step(prediction.float(), t, sample, eta=0.0).prev_sample
        return ((sample - self.action_offset) / self.action_scale)[:, 1:]


@torch.no_grad()
def update_ema(ema, model, step, cfg):
    """先快后慢的 EMA；保存/评测使用 EMA 权重，best 按固定验证噪声 MSE 选取。"""
    decay = min(cfg["ema_decay"], 1 - (1 + step) ** (-cfg["ema_power"]))
    for target, source in zip(ema.parameters(), model.parameters(), strict=True):
        target.lerp_(source.detach(), 1 - decay)
    for target, source in zip(ema.buffers(), model.buffers(), strict=True):
        target.copy_(source)
