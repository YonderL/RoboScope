# SmolVLA：按论文架构重新训练 LIBERO-Spatial

这是独立实验，不覆盖原来的 `smolvla_spatial_seed0`。原实验读取本地 128×128
HDF5 图像，使用 9D 关节状态及 `smolvla_base` 初始化；这里直接使用官方发布的
256×256 图像、8D 末端状态和原生 LeRobot 训练／评测处理器。

“LIBERO 数据都是 128”是不准确的：图像尺寸由数据发布版本和采集设置决定。
本实验固定的 `HuggingFaceVLA/libero` 数据具有两路 256×256 RGB 图像。
提取 Spatial 时保留原始图像字节，不把旧 128 图像插值冒充原生 256 图像。

## 配方及其出处

| 项目 | 本实验 |
| --- | --- |
| 数据 | `HuggingFaceVLA/libero` 中全部 Spatial 轨迹，不额外留出 10%。该固定版本是 10 个任务、432 条轨迹、52970 帧，不是每任务 50 条 |
| 数据版本 | `86958911c0f959db2bbbdb107eb3e17c5f9c798e` |
| VLM | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| VLM 版本 | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |
| 初始化 | 仅加载预训练 VLM；action expert、状态／动作投影随机初始化 |
| 架构 | VLM 前 16 层；expert 宽度为 VLM 的 0.75；冻结 VLM，训练 expert 和状态投影 |
| 观测 | 两路 256 RGB；末端位置 3D + 轴角 3D + 夹爪位置 2D |
| 图像进入模型 | 原生 SmolVLA 保持宽高比，缩放／填充至 512×512 |
| 动作 | 预测 50 步；执行 1 步后重新观测；flow matching 推理 10 步 |
| 更新预算 | 100,000 optimizer steps，有效 batch 64，单张最后一张 RTX 4090 |
| 优化器 | 原生 AdamW：LR 1e-4、betas (0.9, 0.95)、eps 1e-8、weight decay 1e-10、grad clip 10 |
| 调度器 | warmup 1,000；cosine decay 至第 30,000 步的 2.5e-6，之后保持下限 |
| 数值／加速 | BF16 autocast，FP32 主权重；`torch.compile` default 模式 |
| checkpoint | 每 10,000 步及训练末尾保存完整原生 checkpoint |

架构、初始化和 100k/batch64 按 [论文仿真实验](https://arxiv.org/html/2506.01844v1)
对齐。**这不是官方全部实验的逐字复现**：用户要求只训练 Spatial，论文还使用其他
LIBERO suites；论文未公开全部下游调度细节。这里的 warmup1000/decay30000 使用
[官方发布配置](https://huggingface.co/HuggingFaceVLA/smolvla_libero/blob/main/config.json)
及原生 SmolVLA preset，不把预训练章节的 warmup100 当作已证实的下游配置。
发布 checkpoint 的 VLM 全层及 expert 0.5 架构，与论文的 16层／0.75 不同；
本实验显式选择论文架构，不加载该 checkpoint 或 `smolvla_base` 权重。
版本、处理器和这些差异仍可能影响成功率，不能保证达到论文数字。

状态／动作采用 Spatial 子集统计量做 mean/std 归一化。数据端不额外翻转图像；
环境使用 OpenGL 约定，再由原生 `LiberoProcessorStep` 旋转 180°，匹配发布数据。
请不要把旧适配器的垂直翻转、9D joint state 或归一化统计接入这条流程。

## 执行

在项目根目录运行，先选择已安装 LeRobot 及 LIBERO 依赖的解释器：

```bash
export PYTHON=python

# 预览：无下载、无 GPU 操作。
bash scripts/train_smolvla_official_spatial.sh --preview

# 下载固定版本数据、仅提取 Spatial，缓存固定版本 VLM，保存配置。
bash scripts/train_smolvla_official_spatial.sh --stage prepare --start

# 独立输出目录，2 次更新 + 任务0的短评测。batch仍为64，检查真实显存需求。
bash scripts/train_smolvla_official_spatial.sh --stage smoke --start

# 完整训练；完成后自动单卡评测最后一个checkpoint。
bash scripts/train_smolvla_official_spatial.sh --stage train --start

# 中断后继续，恢复模型、optimizer、scheduler、RNG和数据顺序。
bash scripts/train_smolvla_official_spatial.sh --stage train --resume --start

# 仅评测最后一个checkpoint。
bash scripts/train_smolvla_official_spatial.sh --stage evaluate --start
```

下载源数据默认放在 `datasets/libero_official`，提取后的 Spatial 默认放在
`datasets/libero_official_spatial`；可用 `SOURCE_ROOT`、`DATA_ROOT` 覆盖。
模型缓存位于项目 `.cache/smolvla_official`，输出默认位于
`outputs/smolvla_official_spatial_seed0`。`--stage smoke` 自动增加 `_smoke` 后缀。
`--output DIR` 可指定新的正式实验目录。已有训练目录必须显式 `--resume`，源码或
配方变化会被实验契约拒绝，避免混合两个实验。

训练只启动一个 worker，用 UUID 将最后一张 RTX 4090 映射成进程内 `cuda:0`；
`gpu_mapping.json` 保存实际物理卡映射。首次编译可能较慢，smoke必须确认有限loss、
能保存checkpoint且能完成环境评测，再开启正式训练。

## 正式结果与历史记录

合并 MuJoCo 3.3.2 的「ramekin 上的黑碗」50 回合复测后，正式结果是 **448/500（89.6%）**：该任务由 10/50 更新为 47/50，其余九个任务保留原成绩。逐任务对照见 [README](../README.zh-CN.md)，来源与配方见 [评测报告](evaluation_20260926.md)。

下列 411/500 是复测合并前的历史成绩：

100k checkpoint、每次执行 10 步、原生评测：**411/500（82.2%）**。训练中的周期评测是每步重规划，不是这组 10 步执行：20k 为 68%，60k 为 73%，80k 为 72%，100k 为 69%（各 100 回合）。记录在 `results/vla_spatial/snapshot.json`，图由 `python -m roboscope report --study vla` 重绘。

这组数字不能和 ACT/DP 的 500 回合配对比较：图像是 256、状态是 8D 末端、回合上限是 280，原生报告也没有保存 ACT/DP 那套初态编号。只有一个训练 seed。MuJoCo 3.3.2 的补充评测单独记录在[评测报告](evaluation_20260926.md)，已用于正式结果；历史 411/500 记录作为来源保留。

## 评测与查看结果

正式训练每 20,000 步用原生环境评测；独立最终评测也采用论文§4.1的每任务 10 回合，
共 100 回合。随机种子为 0；每次仅一个同步环境、任务顺序评测，避免资产路径跨进程
丢失。原生 Spatial 上限为 280 控制步，reset 后静置 10 步。**这与旧实验的每任务50
回合、600步上限、5步静置和初态／种子分配协议不同**，报告结果时应同时列明协议，
不只比较一个 SR。

训练日志为 `train.log`，独立评测日志为 `evaluate.log`。最终结果位于
`evaluation_final/eval_info.json`，其中 `overall.pc_success` 是百分数，
例如 `90.0` 表示 90%，不是 0.9。原生 checkpoint 位于
`training/checkpoints/<step>/pretrained_model`，`checkpoints/last` 指向最后保存版本。
训练不留出验证集，因此没有按 validation loss 选出的 `best.pt`。
`--skip-eval` 仅跳过脚本末尾的独立评测，不关闭训练中的周期评测。

## CPU 检查

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src "$PYTHON" -m pytest -q tests/test_smolvla_official.py
```

测试覆盖原生配置序列化、论文架构字段、短smoke配置、checkpoint续训／评测路径和
不访问数据或GPU的预览。实际 BF16、编译内存及模拟器行为还需 GPU smoke 验证。
