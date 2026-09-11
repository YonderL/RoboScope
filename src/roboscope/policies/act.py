"""LeRobot ACT with a learned task token; no simulator privileged state is used."""

from collections import deque

import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.act.modeling_act import ACTPolicy, ACTTemporalEnsembler
from torch import nn


class TaskACT(nn.Module):
    """对 LeRobot ACT 的薄封装：任务条件、输入归一化、动作反归一化。

    environment_state 这里只借用 token 接口，输入实际是离散 task_id，
    不包含物体真值位姿等仿真特权信息。它经过可学习 Embedding 成为任务 token，
    经 encoder 后由 decoder cross-attention 访问；无需修改 LeRobot 源码。
    """

    def __init__(self, cfg, manifest, chunk, initialize_backbone=True):
        super().__init__()
        config = ACTConfig(
            chunk_size=chunk,
            n_action_steps=1,
            input_features={
                "observation.state": PolicyFeature(FeatureType.STATE, (9,)),
                "observation.environment_state": PolicyFeature(FeatureType.ENV, (1,)),
                "observation.images.agent": PolicyFeature(FeatureType.VISUAL, (3, 128, 128)),
                "observation.images.wrist": PolicyFeature(FeatureType.VISUAL, (3, 128, 128)),
            },
            output_features={"action": PolicyFeature(FeatureType.ACTION, (7,))},
            dim_model=cfg["dim_model"],
            dim_feedforward=cfg["dim_feedforward"],
            n_encoder_layers=cfg["encoder_layers"],
            n_decoder_layers=cfg["decoder_layers"],
            n_vae_encoder_layers=cfg["vae_layers"],
            kl_weight=cfg["kl_weight"],
            pretrained_backbone_weights=(
                "ResNet18_Weights.IMAGENET1K_V1"
                if cfg["pretrained_backbone"] and initialize_backbone
                else None
            ),
        )
        self.policy = ACTPolicy(config)
        # Reuse the existing ENV token slot. The projection now consumes integer IDs,
        # produces (B,D), and reaches the action decoder via encoder memory.
        self.policy.model.encoder_env_state_input_proj = nn.Embedding(
            len(manifest["tasks"]), cfg["dim_model"]
        )
        for key in ("state_mean", "state_std", "action_mean", "action_std"):
            self.register_buffer(key, torch.tensor(manifest[key], dtype=torch.float32))
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def inputs(self, batch):
        """仅用训练轨迹计算的 mean/std；图像使用 ImageNet mean/std。

        task_id 保持整数，不做连续状态归一化。ACTPolicy 本身不自动替我们
        运行 LeRobot 的 processor，因此所有变换在这个函数中显式完成。
        """
        result = {
            "observation.state": (batch["state"] - self.state_mean) / self.state_std,
            "observation.environment_state": batch["task_id"].long(),
        }
        for src, dest in (("agentview_rgb", "agent"), ("eye_in_hand_rgb", "wrist")):
            result["observation.images." + dest] = (
                batch[src].float() / 255 - self.image_mean
            ) / self.image_std
        if "action" in batch:
            result["action"] = (batch["action"] - self.action_mean) / self.action_std
            result["action_is_pad"] = batch["action_is_pad"]
        return result

    def forward(self, batch):
        return self.policy(self.inputs(batch))

    def loss_tensors(self, batch):
        """与 LeRobot 0.6.1 ACTPolicy.forward 同一个损失公式，但指标不调用 item()。

        item() 会让 CPU 等 GPU；把 epoch 日志需要的指标留在 GPU 上累加，
        最后统一读取。不改变 reconstruction/KL 权重、padding 或反向计算。
        """
        import torch.nn.functional as F

        prepared = self.inputs(batch)
        prepared["observation.images"] = [prepared[k] for k in self.policy.config.image_features]
        predicted, (mu, logvar) = self.policy.model(prepared)
        error = F.l1_loss(prepared["action"], predicted, reduction="none")
        mask = ~prepared["action_is_pad"].unsqueeze(-1)
        l1 = (error * mask).sum() / (mask.sum() * error.shape[-1]).clamp_min(1)
        kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        loss = l1 + self.policy.config.kl_weight * kl
        return loss, {"l1_loss": l1.detach(), "kld_loss": kl.detach()}

    @torch.no_grad()
    def predict(self, batch):
        """推理用 z=0，输出环境动作单位；夹紧范围统一放在 rollout 入口。"""
        return self.policy.predict_action_chunk(self.inputs(batch)) * self.action_std + self.action_mean


class Execution:
    """把预测长度 K 与执行长度 R 分开，避免 TE 消融混入反馈频率变化。

    chunk: 每 K 步预测一次，依次执行全部 K 步；replan: 每步只执行最新首步；
    ensemble: 每步预测，并聚合不同 chunk 对同一环境时刻的预测。
    同一 checkpoint 评测三次，不为 TE 开关重复训练。
    """

    def __init__(self, model, chunk, mode, coefficient):
        self.model, self.chunk, self.mode = model, chunk, mode
        self.queue = []
        self.ensemble = ACTTemporalEnsembler(coefficient, chunk)

    def reset(self):
        # 每个 episode 必须清空，否则上次任务的预测会泄漏到下一次 rollout。
        self.queue.clear()
        self.ensemble.reset()

    def step(self, batch):
        if self.mode == "chunk":
            if not self.queue:
                self.queue = list(self.model.predict(batch).unbind(1))
            return self.queue.pop(0)
        actions = self.model.predict(batch)
        if self.mode == "replan":
            return actions[:, 0]
        if self.mode == "ensemble":
            return self.ensemble.update(actions)
        raise ValueError(self.mode)


class BatchedExecution:
    """槽位独立的执行状态：预测可以合批，但历史绝不能跨 episode 共用。"""

    def __init__(self, model, chunk, coefficient):
        self.model, self.chunk, self.coefficient = model, chunk, coefficient
        self.slots = {}

    def add(self, slot, mode):
        if slot in self.slots:
            raise ValueError("Remove old episode before reusing a slot")
        self.slots[slot] = {
            "mode": mode,
            "queue": deque(),
            "next_action": None,
            "ensemble": ACTTemporalEnsembler(self.coefficient, self.chunk),
        }

    def remove(self, slot):
        del self.slots[slot]

    def needs_prediction(self, slot):
        item = self.slots[slot]
        return item["mode"] != "chunk" or not item["queue"]

    def predict(self, slots, batch):
        predictions = self.model.predict(batch)
        for row, slot in enumerate(slots):
            # Clone：避免一个槽位的 action queue 长期引用整个大 batch 的 storage。
            actions = predictions[row : row + 1].clone()
            item = self.slots[slot]
            if item["mode"] == "chunk":
                item["queue"].extend(actions.unbind(1))
            elif item["mode"] == "replan":
                item["next_action"] = actions[:, 0]
            elif item["mode"] == "ensemble":
                item["next_action"] = item["ensemble"].update(actions)
            else:
                raise ValueError(item["mode"])

    def action(self, slot):
        item = self.slots[slot]
        return item["queue"].popleft() if item["mode"] == "chunk" else item["next_action"]
