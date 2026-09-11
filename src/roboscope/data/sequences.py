"""Temporal windows over exactly the ACT train/validation episode split."""

import numpy as np
import torch

from roboscope.data.libero import FrameDataset
from roboscope.runtime.common import CAMERAS


class SequenceDataset(FrameDataset):
    def __init__(self, manifest, split, image_cache=None):
        super().__init__(manifest, 16, split, image_cache=image_cache)

    def __getitem__(self, index):
        """To=2: observations=[t-1,t], action window=[t-1,...,t+14].

        按 DP 常用时间对齐，生成序列的下标 1 才是当前动作 a[t]。
        episode 开头复制首帧；动作首尾复制边界值并 mask，不跨轨迹。
        抽样锚点仍是 ACT 的全部帧，不因为 Tp 改变样本集合。
        """
        ep = int(np.searchsorted(self.ends, index, side="right"))
        start = int(self.ends[ep - 1]) if ep else 0
        t = index - start
        current = super().__getitem__(index)
        previous = super().__getitem__(max(start, index - 1))
        _, _, _, actions = self.episodes[ep]
        offsets = np.arange(t - 1, t + 15)
        current["action"] = torch.from_numpy(actions[np.clip(offsets, 0, len(actions) - 1)].copy())
        current["action_is_pad"] = torch.from_numpy((offsets < 0) | (offsets >= len(actions)))
        for key in ("state", *CAMERAS):
            current[key] = torch.stack([previous[key], current[key]])
        return current


class StepBatchSampler:
    """按绝对 step 重建固定 shuffle，不依赖预取进度，恢复不重复/跳过样本。

    每个 permutation 周期丢弃不足一个完整 batch 的尾部；下一周期重新洗牌。
    这是批大小不同造成的明确训练差异，不改变轨迹划分。
    """

    def __init__(self, size, batch_size, steps, seed, start_step=0):
        self.size, self.batch, self.steps, self.seed, self.start = size, batch_size, steps, seed, start_step
        self.per_cycle = size // batch_size
        if not self.per_cycle:
            raise ValueError("Dataset smaller than batch")

    def __len__(self):
        return max(0, self.steps - self.start)

    def __iter__(self):
        cycle = None
        for step in range(self.start, self.steps):
            c, position = divmod(step, self.per_cycle)
            if c != cycle:
                order = torch.randperm(
                    self.size, generator=torch.Generator().manual_seed(self.seed + c)
                ).tolist()
                cycle = c
            yield order[position * self.batch : (position + 1) * self.batch]
