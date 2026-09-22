# SmolVLA 的 RLT 后训练

本实现依据 **RL Token: Bootstrapping Online RL with Vision-Language-Action Models**
（[论文](https://arxiv.org/abs/2604.23073)，对应用户提供的 PDF），将其公式 (1)–(5)
和 Algorithm 1 适配到项目的 SmolVLA + LIBERO-Spatial。
这是独立 PyTorch 实现，不是作者官方代码，也不声称复现论文的真实机器人结果。

输入必须是本项目已经 SFT 的 SmolVLA checkpoint。RLT 的输出策略由
**冻结的 SFT SmolVLA + 冻结的 RL token 编码器 + 学习后的 actor** 组成；
在线 RL 不修改 SmolVLA 权重，也不是 PPO 或给 flow loss 加成功率权重。
成功率是否提高，需要实际训练后进行固定初态的配对评测。

学习代码与排查训练问题可读 [RLT 学习笔记](smolvla_rlt_learning_notes.md)：
包含张量形状、梯度路径、稀疏奖励、actor 接管、Q 估值、超参数尺度和日志解读。

## 对应论文的两阶段训练

1. **RL token 重建训练**：从冻结 SmolVLA 最终 VLM 层（含最终归一化）只选取图像位置的 token，
   排除语言和状态位置，与论文图 2、脚注 1 的实验设置对应。
   小型 Transformer encoder 在输入末尾追加一个可学习 readout，输出单个 RL token。
   因果 decoder 仅看到 RL token 和右移后的、stop-gradient 的原始 embeddings，
   自回归重建这些图像特征。decoder 不能访问整个 encoder 序列，不能看到当前或未来 target。
   这一步使用原有 SFT manifest 的训练演示，采用论文允许的 `alpha=0`，保持 SFT 完全冻结。
2. **在线 actor/critic**：冻结 RL token 与 SmolVLA，用 LIBERO rollout 填充 replay。
   critic 使用 chunk TD 目标，actor 直接生成动作块，优化 Q 值和到 VLA 参考动作的 L2 距离。
   actor 不是残差网络。每两次 critic 更新进行一次 actor 更新，target critic 用 Polyak 更新。

```text
LIBERO RGB / language / proprio
             |
        frozen SmolVLA
         /          \
final image tokens  reference actions [50, 7]
       |                    |
frozen RL-token encoder     first 10 actions
       |                    |
       +-- state + reference --> actor --> LIBERO rollout
                                      |
                              rollout-only replay
                                      |
                            twin critic / actor updates
```

SmolVLA 的 prefix KV cache 同时用于最终特征读取与原生 flow sampling，避免重复运行 VLM。
同一噪声下，适配器的参考动作已通过与原生 SmolVLA 输出逐元素一致的回归测试。
图 2 仍把 prompts 输入 VLA；脚注 1 的省略语言 embeddings 指 RL token 重建阶段。
本实现同样保留冻结 SmolVLA 的完整图像/语言/状态前向和 KV cache，只在重建分支选取图像位置。
图像特征可能已通过 VLM 注意力融合语言信息，这与直接把语言位置送进重建网络不同。
9D 本体状态在提取 RL token 后单独归一化、拼接，供 actor 和 critic 使用，不参与重建目标。

## Replay buffer 与 TD 目标

Replay **只接受本轮 LIBERO rollout**：冻结 SFT 的 warmup 轨迹和在线 actor 轨迹。
原始 HDF5 演示仅用于 token 重建，不注入 RL replay；不筛掉失败轨迹。
每个控制步的观测和实际执行动作先记录在当前 episode 中，再每隔 2 步构造一个窗口：

```text
(x_t, executed_actions[t:t+C], reference_t,
 discounted_reward, bootstrap_discount, x_next, reference_next,
 action_mask, duration, terminated, truncated,
 task_id, episode_id, control_step, seed, warmup)
```

`x` 是 RL token、归一化的 9D joint/gripper 状态、剩余回合时间的拼接。
由于 VLA 和 token 在 RL 阶段冻结，buffer 保存这些紧凑向量即可；无需反复运行 VLA，也无需长期保存 RGB。
对跨越推理调用边界的 stride-2 窗口，使用实际连续执行的动作，
并为中间状态重新生成对齐的 VLA reference，不使用前一次 reference 的错位切片。

设该窗口实际执行 `n <= C` 步：

```text
R = sum_{j=0}^{n-1} gamma^j * reward[t+j]
target_Q = R + discount * min(Q1_target(x_next, a_next), Q2_target(x_next, a_next))
discount = 0                          if success or finite-horizon deadline
           gamma^n                    otherwise
a_next ~ current_actor(x_next, reference_next)
actor_loss = mean(-min(Q1, Q2) + beta * sum((sampled_action - reference)^2))
```

成功仅由 LIBERO `check_success()` 给出：首次成功的控制步奖励为 1，其余为 0。
短尾补零并保存有效动作 mask；critic 输入和 actor 的正则只使用实际执行部分。
达到 600 步视为此基准任务的有限时域终止，记录 `truncated=true` 并停止 bootstrap。
因此状态显式包含剩余时间。若未来改成无限时域任务，这项处理必须相应改变。

Reference dropout 在 actor 更新时以 50% 概率将整条参考动作输入置零；
L2 正则的目标仍是原始 reference，不是置零后的输入。目标 Q 和 rollout 时不做 reference dropout。
actor 是固定标准差的 Gaussian；均值用 tanh 限制，采样动作裁剪到 LIBERO 的 [-1,1]。
评测使用 actor 均值，VLA reference 仍使用每回合独立、固定种子的 flow noise。

## 默认配置与适配差异

配置：`configs/libero_spatial/smolvla_rlt.json`。

| 项目 | 本实现 | 来源 |
|---|---|---|
| VLA 预测块 / RL 执行块 | 50 / **10** | 论文的 H / C；原 SFT 配置仍执行 50 |
| RL token 输入 / 重建目标 | 最终 VLM 层的图像位置特征 | 图 2、脚注 1；`token_features=vlm_image_tokens` |
| Replay stride | 2 | 论文 |
| Reference dropout | 0.5 | 论文附录 |
| Twin Q、critic:actor 更新比 | 2 个 Q，2:1 更新 | 论文 |
| 每新增 replay transition 的 critic 更新数 | 5 | 按论文 UTD=5 做出的显式计数约定 |
| Actor / critic MLP | 两层 256，LayerNorm + ReLU | 层数/宽度参考论文；归一化是工程选择 |
| Token 重建更新数 | 2,000 | 论文给出 2,000–10,000 |
| Token encoder / decoder | 各 2 层，宽 256，8 heads | SmolVLA 适配值；论文图示 readout 宽 2,048 |
| Token loss | 有效 token/feature 的平均 MSE | 公式 (2) 的缩放实现 |
| Token batch / lr | 32 / 1e-4 | 工程默认值 |
| Actor / critic lr | 3e-4 / 3e-4 | 工程默认值，论文未提供具体数值 |
| Actor std / beta | 0.05 / 0.1 | 工程默认值；beta 对应 **L2 平方和** |
| gamma / target tau | 0.99 / 0.005 | 工程默认值 |
| Replay / RL batch | 50,000 transitions / 256 | 工程默认值 |
| Warmup / online budget | 6,000 / 100,000 控制步 | 工程预算；online budget 包括 warmup |
| Warmup 后额外预更新 | 1,000 次 critic 更新，含延迟 actor 更新 | 避免直接用未训练 actor 接管的工程选择 |
| 采集/学习调度 | 单环境，完整 episode 后同步更新 | 为可读性和精确恢复采用同步实现；论文采用异步 |
| 控制频率 / proprio | 20 Hz / 9D joint+gripper | 沿用项目；论文实机是 50 Hz，并使用额外速度状态 |

训练任务按 manifest 的 10 个任务轮转。训练环境调用带独立种子的随机 `reset()`，
不会加载正式评测的固定初态文件；正式评测始终使用官方固定初态 0–49。
保存完整 episode 后才提交更新与 checkpoint，最终预算最多超过 `online_steps` 一个 episode。
论文中的人工干预、关键阶段切换未加入本仿真版本；奖励由环境自动提供。

这是可运行的起点配置，不能将未公开的超参数描述为“论文原值”。
`rollouts.json` 记录训练成功率和 `replay_rewarded_transitions`；若始终没有成功奖励，
说明当前 SFT 的探索数据没有提供成功信号，应先检查 SFT 表现、任务重置分布或增加采集预算。
训练成功率不是正式评测分数，也不用于选取测试集上的 best checkpoint。

## 运行

推荐顺序：**SFT → RL token 重建并冻结 → SFT warmup rollout → 在线 RL → 配对评测**。
当前 replay 缓存的是 token 向量；若先用未训练 token 采集，再更新 token，已有向量会失效。
并行进行“原始 RGB rollout 采集”和 token 训练理论上可行，但需要额外的原始轨迹存储及重编码阶段，
本实现未提供。在线 RL 阶段会持续采集 actor 的新轨迹并更新 replay，不是在固定 buffer 上只做离线训练。

**Warmup 采集是什么？** 让已有的 SFT 策略先在 LIBERO 中执行，收集第一批状态、真实动作、奖励、
下一状态；成功和失败都保存。此时不让随机初始化的 actor 接管。独立 warmup 阶段只采集，
进入 online 阶段后才先进行接管前更新，随后改由 actor 采集。这里的 warmup 是经验采集，
与学习率从小到大增长的 warmup 无关。

Replay 并不必须保存 RL token。另一种设计是保存原始 RGB/本体状态，在训练 batch 中实时提取特征；
这种设计允许先采集再训练 token，但反复抽样时会重复运行大模型。
当前采用冻结特征缓存来节省计算，因此有“token 先冻结”的依赖。不要同时用两个进程改写同一 run 的 buffer。

复用已有的 `requirements-training.txt`、`requirements-smolvla.txt` 环境，无额外 RL 框架依赖。
首先完成 [SmolVLA SFT](smolvla_spatial.md)，确保 SFT 输出中已有 `final.pt` 或 `best.pt`。
**当前项目目录尚无已完成的 SmolVLA SFT 权重，不能直接从空目录开始 RLT。**

### 按阶段执行

在项目根目录、安装好依赖的环境中运行。`DATA_ROOT` 必须包含 `libero_spatial/`，
`LIBERO_ROOT` 必须包含 `assets/`、`bddl_files/`、`init_files/`。脚本默认路径是项目内
`datasets/` 与 `LIBERO/libero/libero/`，外置数据请通过环境变量指定。

```bash
conda activate lerobot
export PYTHON="$(command -v python)"
export SFT_SOURCE="$PWD/outputs/smolvla_spatial_seed0"

# 1. SFT；默认训练结束后评测原生 C=50、500 回合
bash scripts/train_smolvla_spatial.sh --output "$SFT_SOURCE" --start

# 以下阶段共用同一个 RLT 输出目录和 recipe；应在开始 token 阶段前确定所有超参数。
export OUTPUT_ROOT="$PWD/outputs/smolvla_rlt_spatial_seed0"

# 2. 重建训练；只生成 token.pt/token_last.pt，不创建 LIBERO 环境
bash scripts/posttrain_smolvla_rlt_spatial.sh --stage token --start

# 3. 用冻结 SFT 在训练随机初态采集 warmup；保存 last.pt 中的 replay，不更新 actor/critic
bash scripts/posttrain_smolvla_rlt_spatial.sh --stage warmup --start --resume

# 4. 先执行接管前更新，再持续采集 actor rollout + 更新网络；生成 final.pt
bash scripts/posttrain_smolvla_rlt_spatial.sh --stage online --start --resume --skip-eval

# 5. 分别评测 RLT C=10 和冻结 SFT C=10，每种各 500 回合
bash scripts/posttrain_smolvla_rlt_spatial.sh --stage evaluate --start
```

`--resume` 在步骤 3/4 表示继续同一个运行目录，适用于正常阶段切换，也适用于中断恢复。
阶段入口会检查前置产物，缺少 token 或 warmup checkpoint 时明确报错。
warmup 以完成整回合为边界，达到 `warmup_steps` 且 replay 至少有一个 batch 后停止；
若采样数不足会继续 SFT 采集。warmup 重复执行 `--resume` 不会重复采集已经完成的部分。
`initial_updates` 在步骤 4 执行，不能把 warmup 结束时保存的随机 actor 当作训练后的策略。
一键运行与分阶段恢复的参数、replay 和更新计数已做一致性回归检查。

日志分别为 `train_token.log`、`train_warmup.log`、`train_online.log`，一键运行使用 `train_all.log`。
`last.pt` 的 `replay` 字段保存 buffer；没有另外导出一份脱离 token 身份的通用 replay 文件。
默认 token 预算仍为 2000 更新。若采用建议的 5000 更新，请先复制 recipe 并修改 `token_steps`，
用同一个 `RECIPE` 环境变量贯穿所有阶段；不能直接修改旧 run 配置后强行 resume。

### 一键运行与恢复

```bash
# 默认只预览，不加载权重、不创建训练目录
bash scripts/posttrain_smolvla_rlt_spatial.sh --source outputs/smolvla_spatial_seed0 --preview

# token 重建 -> warmup -> 在线 RL -> RLT 和同执行长度 SFT 的正式评测
bash scripts/posttrain_smolvla_rlt_spatial.sh --source outputs/smolvla_spatial_seed0 --start

# 中断恢复；保留相同配置、源码和 SFT checkpoint
bash scripts/posttrain_smolvla_rlt_spatial.sh --source outputs/smolvla_spatial_seed0 --start --resume
```

脚本支持 `PYTHON`、`SFT_SOURCE`、`OUTPUT_ROOT`、`RECIPE` 环境变量和
`--stage all|token|warmup|online|evaluate`、`--output`、`--checkpoint best|final`、`--skip-eval`。
使用 `--start --smoke` 可在独立 `_smoke` 目录运行：token 2 updates、在线预算 120 步、
每回合最多 20 步、每任务评测 1 回合。这只用于运行检查，不是性能评测。

也可以分开执行：

```bash
python -m roboscope posttrain \
  --source outputs/smolvla_spatial_seed0 \
  --recipe configs/libero_spatial/smolvla_rlt.json \
  --output outputs/smolvla_rlt_spatial_seed0 --start

# RLT actor，C=10，500 回合
python -m roboscope evaluate --source outputs/smolvla_rlt_spatial_seed0 \
  --output outputs/smolvla_rlt_spatial_seed0/evaluation_rlt --episodes 50 --start

# 冻结 SFT 的同长度对照，C=10，另一个独立的 500 回合评测
python -m roboscope evaluate --source outputs/smolvla_rlt_spatial_seed0 \
  --output outputs/smolvla_rlt_spatial_seed0/evaluation_sft_c10 \
  --episodes 50 --rlt-reference --start
```

建议同时保留原 SFT C=50 的评测。判断 RL 本身的增益，应首先比较 RLT C=10 与 SFT C=10；
仅比较 RLT C=10 与 SFT C=50，会混入重规划频率变化。
三个评测都保持项目的 10 任务、每任务 50 固定初态、600 步、5 步静置、
相同相机/种子/控制器/动作裁剪/成功判定、每 GPU 8 个环境和 5+30 次延迟测量。
每个评测有独立的输出目录与 checkpoint hash，评测轨迹不回流进训练 replay。

## 代码与恢复

| 文件 | 职责 |
|---|---|
| `policies/smolvla_rlt.py` | 加载冻结 SFT，原生 prefix/cache/flow 适配，RLT 推理 |
| `rl/token.py` | Readout encoder 和因果 reconstruction decoder |
| `rl/networks.py` | Gaussian actor 与 twin Q |
| `rl/learner.py` | TD 目标、actor 正则、reference dropout、target 更新 |
| `rl/replay.py` | Ring buffer、stride-2 chunk、奖励与终止处理 |
| `rl/collector.py` | LIBERO rollout、中间状态特征与 reference 对齐 |
| `trainers/smolvla_rlt.py` | 两阶段训练、度量和断点恢复 |
| `workflows/smolvla_rlt.py` | SFT 来源核验、目录/配置、GPU 启动 |

`token_last.pt` 保存 token 阶段优化器和 RNG，`token.pt` 保存已完成的重建模型。
在线 `last.pt` 原子保存 actor、critic、target critic、优化器、完整 replay、
replay 采样 RNG、PyTorch/Python/NumPy RNG 和已提交的 episode 编号。
重启会重做未提交 episode，不混入半条轨迹。模拟器异常会中止，不计成策略失败。
`final.pt` 保存 token/actor 和冻结 SFT 的路径、SHA256 引用，不重复复制大模型。
推理仍需要保留原 SFT run 及其 VLM 配置/tokenizer 资源。

早期版本曾将完整 prefix（图像、语言、状态）用于重建；现已更正为仅图像位置。
新配置使用 `token_features=vlm_image_tokens` 标识表示语义，拒绝直接载入缺少该标识的旧 RLT 模型。
旧版 token、actor/critic 和 replay 不能通过补填字段继续使用，需要在新目录重新进行两个 RLT 阶段；
原 SFT checkpoint 不受影响。

## 验证范围

完整回归测试通过（71 tests）；Ruff、全项目 Python 格式检查、分阶段脚本预览和源码打包通过。
自动测试覆盖因果重建无 target 泄露、梯度不流入 SFT、真实微型 SmolVLA 的参考动作一致性、
仅图像位置的提取、语言/state/padding 排除、下游本体状态拼接、旧表示配置拒绝加载，
Gaussian/双 Q/延迟 actor 更新、reference dropout、chunk 奖励与实际时长、成功/超时终止、
跨推理边界窗口、replay 覆盖与恢复、token/在线两阶段恢复，以及原有 SFT 50 步和 RLT 10 步评测队列。
修正后重新运行的真实 GPU/LIBERO 短测试使用微型 SmolVLA 权重，完成了两个 12 步回合（先 SFT warmup、再 actor），
生成 12 条 replay transitions，warmup 独立保存时更新次数为 0，恢复在线训练后累计执行 14 次 learner 更新，
输出均有限；这验证了阶段切换、replay 恢复和采集到更新的实际链路。
完整 SFT 权重的 RLT 训练和 500 回合增益测量尚未进行，不宣称已提高成功率。
