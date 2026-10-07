"""RLT equations (1)-(2): learned readout and causal embedding reconstruction."""

import math

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


def positions(length, width, device):
    indices = torch.arange(length, device=device).float().unsqueeze(1)
    rates = torch.exp(torch.arange(0, width, 2, device=device).float() * (-math.log(10000) / width))
    result = torch.zeros(length, width, device=device)
    result[:, 0::2], result[:, 1::2] = (indices * rates).sin(), (indices * rates).cos()
    return result


class RLToken(nn.Module):
    """Decoder sees only the bottleneck and shifted, detached teacher tokens.

    It cannot cross-attend to the original encoder sequence. Otherwise a
    reconstruction shortcut would defeat the learned RL-token bottleneck.

    Production RLT keeps width == feature_dim (SmolVLM hidden size) so image
    embeddings enter the readout transformer without a down-projection. A
    narrower width with Linear projections remains only for tiny unit tests.
    """

    def __init__(self, feature_dim, width=960, layers=2, heads=8):
        super().__init__()
        # 新增的 readout 网络独立于冻结 SmolVLA；不在 VLM 内部插入可训练层。
        # 默认 encoder 2 层 + decoder 2 层，两次构造产生独立参数，不共享权重。
        # feature_dim 来自 VLM；width 是 readout 内部宽度及 RL token 维度。
        # 生产路径要求 width == feature_dim，直接使用 VLM embedding，不做降维投影。
        self.feature_dim = feature_dim
        self.width = width
        self.input = None if width == feature_dim else nn.Linear(feature_dim, width)
        self.readout = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.encoder = self._transformer(width, layers, heads)
        self.decoder_input = None if width == feature_dim else nn.Linear(feature_dim, width)
        self.decoder = self._transformer(width, layers, heads)
        # 论文公式 (2) 的线性输出投影 h_φ；宽与特征维相同时也保留。
        self.output = nn.Linear(width, feature_dim)

    @staticmethod
    def _transformer(width, layers, heads):
        # 每个 block：pre-LayerNorm -> 多头自注意力 -> 残差 -> pre-LayerNorm
        # -> FFN(GELU) -> 残差；默认 8 heads、FFN 宽 4*width、dropout=0。
        # encoder 和 decoder 都用 TransformerEncoder 搭建；decoder 在 reconstruct
        # 中加入 causal mask，成为仅自注意力的因果重建网络，没有 cross-attention。
        # 最后的 LayerNorm 不计作额外 Transformer 层。
        layer = nn.TransformerEncoderLayer(
            width, heads, width * 4, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        return nn.TransformerEncoder(layer, layers, norm=nn.LayerNorm(width), enable_nested_tensor=False)

    def _project(self, projection, features):
        return features if projection is None else projection(features)

    def encode(self, features, valid):
        # [B, N_image, D_vlm] -> [B, token_dim]；detach 禁止重建损失更新 SFT。
        # features 仅含 VLM 最终层的图像位置，由 prefix_prefill 排除语言/state 位置。
        # readout 读取所有有效图像 token；本体状态在下游 actor/critic 输入处再拼接。
        features = features.detach().float()
        sequence = torch.cat(
            [self._project(self.input, features), self.readout.expand(len(features), -1, -1)], dim=1
        )
        sequence = sequence + positions(sequence.shape[1], self.width, sequence.device)
        padding = torch.cat(
            [~valid.bool(), torch.zeros(len(features), 1, device=valid.device, dtype=torch.bool)], dim=1
        )
        # PyTorch 2.7.1 / Blackwell FP32 efficient SDPA produced incorrect
        # gradients at the production head width (120). Math SDPA agrees with
        # an FP64 reference; restrict this workaround to the trainable token.
        with sdpa_kernel(SDPBackend.MATH):
            return self.encoder(sequence, src_key_padding_mask=padding)[:, -1]

    def reconstruct(self, token, targets, valid):
        # [z_rl, z_0, ..., z_(M-2)] predicts [z_0, ..., z_(M-1)].
        # 右移输入 + 因果 mask：预测 z_i 时只能看到 readout 和 z_<i。
        # 不提供 encoder 全序列的 cross-attention，避免绕过单 token 瓶颈。
        teacher = self._project(self.decoder_input, targets.detach().float()[:, :-1])
        shifted = torch.cat([token[:, None], teacher], dim=1)
        length = shifted.shape[1]
        shifted = shifted + positions(length, self.width, shifted.device)
        padding = torch.cat(
            [torch.zeros(len(valid), 1, device=valid.device, dtype=torch.bool), ~valid[:, :-1].bool()], dim=1
        )
        causal = torch.ones(length, length, device=shifted.device, dtype=torch.bool).triu(1)
        with sdpa_kernel(SDPBackend.MATH):
            decoded = self.decoder(shifted, mask=causal, src_key_padding_mask=padding)
        return self.output(decoded)

    def forward(self, features, valid):
        # 重建误差衡量特征压缩，不衡量任务成功率；loss 小不保证状态足够用于控制。
        token = self.encode(features, valid)
        prediction = self.reconstruct(token, features, valid)
        error = (prediction - features.detach().float()).square().mean(-1)
        return (error * valid).sum() / valid.sum().clamp_min(1)
