"""Small direct-action Gaussian actor and twin Q functions (RLT equations 3-5)."""

import torch
from torch import nn


def mlp(input_dim, output_dim, hidden):
    layers = []
    for width in hidden:
        layers.extend([nn.Linear(input_dim, width), nn.LayerNorm(width), nn.ReLU()])
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, state_dim, chunk_size, hidden=(256, 256), std=0.05):
        super().__init__()
        # 论文 Eq.(4): N(mu_theta(x, reference), sigma^2 I)；附录 B 明确 sigma 固定。
        # 因此网络只预测均值，不另设可学习方差头；0.05 是本项目默认值，非论文公布值。
        self.chunk_size, self.std = chunk_size, std
        self.net = mlp(state_dim + chunk_size * 7, chunk_size * 7, hidden)

    def forward(self, state, reference, dropout=0.0):
        # state: [B, token_dim + 9 + 1]；reference: [B, C, 7]。
        # 每个样本整条 reference 一起丢弃，迫使 actor 也利用状态判断动作。
        if dropout:
            keep = torch.rand(len(state), 1, 1, device=state.device) >= dropout
            reference = reference * keep
        mean = self.net(torch.cat([state, reference.flatten(1)], dim=-1))
        return mean.tanh().reshape(-1, self.chunk_size, 7)

    def sample(self, state, reference, dropout=0.0, generator=None):
        mean = self(state, reference, dropout)
        noise = torch.randn(mean.shape, device=mean.device, generator=generator)
        # 重参数化：噪声本身不求导，但 mean -> action -> Q 的梯度链仍存在。
        # clamp 后是裁剪的 Gaussian，不再是严格 Gaussian；越界部分梯度为零。
        # The mean is direct action output, not a residual added to the VLA.
        # Clip Gaussian samples to LIBERO's actual actuator bounds.
        return (mean + self.std * noise).clamp(-1, 1)


class TwinQ(nn.Module):
    def __init__(self, state_dim, chunk_size, hidden=(256, 256)):
        super().__init__()
        self.q1 = mlp(state_dim + chunk_size * 7, 1, hidden)
        self.q2 = mlp(state_dim + chunk_size * 7, 1, hidden)

    def forward(self, state, action):
        # 每个 Q 评价整段 C 步动作的折扣回报，输出 [B, 2]，不是逐步动作分数。
        inputs = torch.cat([state, action.flatten(1)], dim=-1)
        return torch.cat([self.q1(inputs), self.q2(inputs)], dim=-1)
