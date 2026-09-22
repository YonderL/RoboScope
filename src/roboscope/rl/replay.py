"""Bounded rollout-only replay, with explicit chunk durations and provenance."""

import numpy as np
import torch


class ReplayBuffer:
    # 环形数组只保留最近 capacity 条 transition；容量单位不是 episode，也不是控制步。
    # 特征提取器在在线阶段被冻结，所以可以缓存向量而不用每次采样重新处理 RGB。
    def __init__(self, capacity, seed=0):
        self.capacity, self.cursor, self.size = capacity, 0, 0
        self.arrays = {}
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.size

    def add(self, row):
        row = {key: np.asarray(value) for key, value in row.items()}
        if not self.arrays:
            self.arrays = {
                key: np.empty((self.capacity, *value.shape), dtype=value.dtype) for key, value in row.items()
            }
        if row.keys() != self.arrays.keys():
            raise ValueError("Replay schema changed")
        for key, value in row.items():
            if value.shape != self.arrays[key].shape[1:] or not np.isfinite(value).all():
                raise ValueError(f"Invalid replay field {key}")
            self.arrays[key][self.cursor] = value
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, count, device):
        if not self.size:
            raise ValueError("Cannot sample empty replay")
        # 均匀、有放回采样；不按任务或成功率平衡，长失败回合可能占据更多样本。
        indices = self.rng.integers(self.size, size=count)
        return {key: torch.from_numpy(value[indices]).to(device) for key, value in self.arrays.items()}

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "cursor": self.cursor,
            "size": self.size,
            "arrays": {key: value[: self.size].copy() for key, value in self.arrays.items()},
            "rng": self.rng.bit_generator.state,
        }

    def load_state_dict(self, state):
        if state["capacity"] != self.capacity:
            raise ValueError("Replay capacity changed on resume")
        self.cursor, self.size = state["cursor"], state["size"]
        self.arrays = {
            key: np.empty((self.capacity, *value.shape[1:]), dtype=value.dtype)
            for key, value in state["arrays"].items()
        }
        for key, value in state["arrays"].items():
            self.arrays[key][: self.size] = value
        self.rng.bit_generator.state = state["rng"]


def episode_transitions(actions, success, features, cfg, task_id, episode_id, seed, warmup):
    """Sliding windows every stride steps, never across reset boundaries.

    features maps control-step index to (RL state, VLA reference). Only actual
    simulator-executed, clipped actions enter replay. Both success and the
    benchmark's finite 600-step deadline are terminal, with no bootstrap.
    """
    actions = np.asarray(actions, dtype=np.float32)
    length, chunk, gamma = len(actions), cfg["action_horizon"], cfg["gamma"]
    if not length:
        raise ValueError("Cannot store an empty rollout")
    for start in range(0, length, cfg["replay_stride"]):
        # C=10、stride=2 时窗口为 [0:10]、[2:12]……；相邻窗口高度相关。
        # 这里使用真实执行动作，可以跨预测边界，但不能跨 episode/reset。
        end = min(start + chunk, length)
        duration = end - start
        terminal = end == length
        state, reference = features[start]
        if terminal:
            # 当前任务把成功和时间耗尽都视为终点，discount=0，next_* 仅作占位。
            # 这是有限时域建模选择；不能直接套用于需要超时 bootstrap 的任务。
            next_state, next_reference = np.zeros_like(state), np.zeros_like(reference)
        else:
            next_state, next_reference = features[end]
        executed = np.zeros((chunk, 7), np.float32)
        executed[:duration] = actions[start:end]
        # 环境只在最后一次成功动作给奖励 1；因此成功窗口的 R=gamma**(n-1)。
        # 更早的窗口奖励仍为 0，依赖 TD bootstrap 将成功信号逐渐向前传播。
        yield {
            "state": np.asarray(state, np.float32),
            "next_state": np.asarray(next_state, np.float32),
            "reference": np.asarray(reference, np.float32),
            "next_reference": np.asarray(next_reference, np.float32),
            "action": executed,
            "action_mask": np.arange(chunk) < duration,
            "reward": np.float32(gamma ** (duration - 1) if terminal and success else 0),
            "discount": np.float32(0 if terminal else gamma**duration),
            "terminated": np.bool_(terminal and success),
            "truncated": np.bool_(terminal and not success),
            "duration": np.int64(duration),
            "task_id": np.int64(task_id),
            "episode_id": np.int64(episode_id),
            "control_step": np.int64(start),
            "seed": np.int64(seed),
            "warmup": np.bool_(warmup),
        }
