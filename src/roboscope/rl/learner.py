"""Off-policy RLT updates. No entropy bonus, PPO ratio, residual policy or VLA gradients."""

import copy

import torch
from torch import nn

from roboscope.rl.networks import Actor, TwinQ


def bellman_target(reward, discount, next_q):
    """discount is zero at episode end, otherwise gamma ** actual executed steps."""
    return reward + discount * next_q


class RLTAgent(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        state_dim = cfg["token_dim"] + 9 + 1  # token, normalized proprio, remaining time
        self.actor = Actor(state_dim, cfg["action_horizon"], cfg["hidden_dims"], cfg["actor_std"])
        self.critic = TwinQ(state_dim, cfg["action_horizon"], cfg["hidden_dims"])
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg["actor_lr"])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg["critic_lr"])
        self.updates = 0

    def update(self, batch):
        # 阅读顺序：固定 TD 标签 -> 拟合 critic -> 延迟更新 actor -> 慢更新 target。
        # replay 中的 state/reference 已脱离 VLA 计算图，这里只训练小网络。
        cfg = self.cfg
        with torch.no_grad():
            # y = R_chunk + gamma**n * min(Q1_target, Q2_target)。
            # no_grad 把右侧当作监督标签，避免 critic 通过改变标签来降低误差。
            # Equation (3) uses the current stochastic policy and target critics.
            # Reference dropout is only an actor-training regularizer.
            next_action = self.actor.sample(batch["next_state"], batch["next_reference"])
            next_q = self.target_critic(batch["next_state"], next_action).amin(-1)
            target = bellman_target(batch["reward"], batch["discount"], next_q)
        action_mask = batch["action_mask"].unsqueeze(-1)
        # episode 提前结束时，尾部不足 C 步；补零只是占位，不能当成执行过的动作。
        q = self.critic(batch["state"], batch["action"] * action_mask)
        critic_loss = (q - target[:, None]).square().mean()
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
        self.critic_optimizer.step()
        self.updates += 1
        metrics = {
            "critic_loss": critic_loss.item(),
            "q": q.detach().mean().item(),
            "target_q": target.mean().item(),
            "updates": self.updates,
        }
        if self.updates % cfg["policy_delay"] == 0:
            # 冻结 Q 的参数，但保留 Q 对 action 的导数；不能用 no_grad 包裹此分支。
            # 这样 actor 能沿 Q 的梯度改进动作，critic 权重却不会被 actor loss 修改。
            self.critic.requires_grad_(False)
            try:
                action = self.actor.sample(batch["state"], batch["reference"], cfg["reference_dropout"])
                value = self.critic(batch["state"], action * action_mask).amin(-1)
                # Eq. (5) uses squared L2 SUM, not mean over chunk coordinates.
                penalty = ((action - batch["reference"]).square() * action_mask).sum((1, 2))
                # 最小化 -Q 鼓励高回报；+ beta*L2 抑制偏离 SFT 的动作。
                # dropout 只修改 actor 输入，正则目标仍用未丢弃的 reference。
                actor_loss = (-value + cfg["bc_weight"] * penalty).mean()
                self.actor_optimizer.zero_grad(set_to_none=True)
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), cfg["grad_clip"], error_if_nonfinite=True
                )
                self.actor_optimizer.step()
                metrics.update(actor_loss=actor_loss.item(), bc_penalty=penalty.mean().item())
            finally:
                self.critic.requires_grad_(True)
        with torch.no_grad():
            # target <- (1-tau)*target + tau*critic，减缓 TD 标签的变化。
            # target 网络不经 optimizer 更新；双 Q 取最小值也不能保证估值准确。
            for target_parameter, parameter in zip(
                self.target_critic.parameters(), self.critic.parameters(), strict=True
            ):
                target_parameter.lerp_(parameter, cfg["target_tau"])
        return metrics

    def training_state(self):
        return {
            "model": self.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "updates": self.updates,
        }

    def restore(self, state):
        self.load_state_dict(state["model"], strict=True)
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.updates = state["updates"]
