# RoboScope

**以可复现的闭环评测为核心，研究视觉机器人策略。**

[English](README.md) · [环境与复现命令](docs/quickstart.md) · [架构](docs/architecture.md) · [实验配置](docs/experiments.md) · [研究分析](docs/findings.md)

当前实现：LIBERO-Spatial 上的多任务 ACT、Diffusion Policy，以及官方数据上的 SmolVLA 和 HF Spatial 上的 Pi-0 LoRA。下面按任务名对齐四个方法的闭环成功率。ACT/DP 用 HDF5 任务顺序，SmolVLA 用原生顺序（「ramekin 上的黑碗」是原生 T5、HDF5/Pi-0 的 T7）。

![四种策略逐任务正式成功率](docs/assets/policy_success_rates.png)

## 正式闭环结果（2026-09-28）

SR = 成功回合数 / 评测回合数。每格列出 SR 和成功次数，每任务 50 回合、每个模型共 500 回合；四个模型均为训练 seed 0。任务均为抓取对应位置的黑碗并放到盘子上。

| 任务 | ACT | DP | SmolVLA | Pi-0 LoRA |
|---|---:|---:|---:|---:|
| 盘子和 ramekin 之间 | 84% (42/50) | 96% (48/50) | 94% (47/50) | 100% (50/50) |
| 桌面中央 | 96% (48/50) | 100% (50/50) | 98% (49/50) | 98% (49/50) |
| 木柜顶层抽屉 | 90% (45/50) | 90% (45/50) | 88% (44/50) | 92% (46/50) |
| 饼干盒旁边 | 96% (48/50) | 94% (47/50) | 98% (49/50) | 94% (47/50) |
| 盘子旁边 | 76% (38/50) | 70% (35/50) | 76% (38/50) | 74% (37/50) |
| ramekin 旁边 | 82% (41/50) | 98% (49/50) | 94% (47/50) | 84% (42/50) |
| 饼干盒上面 | 92% (46/50) | 94% (47/50) | 90% (45/50) | 90% (45/50) |
| ramekin 上面 | 68% (34/50) | 74% (37/50) | 94% (47/50) | 84% (42/50) |
| 炉子上面 | 86% (43/50) | 88% (44/50) | 84% (42/50) | 76% (38/50) |
| 木柜上面 | 98% (49/50) | 84% (42/50) | 80% (40/50) | 84% (42/50) |
| **全套** | **434/500（86.8%）** | **444/500（88.8%）** | **448/500（89.6%）** | **438/500（87.6%）** |

ACT 是 K=8、epoch 32 的 chunk 执行。DP 是 30k final，DDIM=10、Ta=8。SmolVLA 是 100k 官方 Spatial checkpoint，每次执行 10 步。Pi-0 是 30k HF Spatial LoRA，每次执行 8 步，十个任务都在 MuJoCo 3.3.2 上评测。

「ramekin 上面」这一行，ACT、DP、SmolVLA 用的是 PRO 5000 上 MuJoCo 3.3.2 的复测；它们另外九个任务仍是此前已发布的成绩。图像尺寸、本体状态和回合上限并不相同，这张表是合并复测后的正式结果；不同方法仍使用各自的观测与执行协议。逐任务来源、checkpoint 指纹见 [正式结果](results/official_spatial/summary.json)，复测配置见 [protocol.json](results/mujoco332/protocol.json)。

## 实验进展

四个模型的训练与每任务 50 回合评测均已完成：ACT K8 为 epoch 32（56,064 更新），DP final EMA 为 30k 更新，SmolVLA 为 100k 更新，Pi-0 LoRA HF Spatial 为 30k 更新。ramekin 任务复测由 ACT 30% → 68%、DP 4% → 74%、SmolVLA 20% → 94%，已纳入上表；Pi-0 LoRA 该任务为 84%。

对应配方：[ACT](configs/libero_spatial/act_baseline.json) · [DP](configs/libero_spatial/diffusion.json) · [SmolVLA](configs/libero_spatial/smolvla_official.json) · [Pi-0 LoRA](configs/libero_spatial/pi0_lora_hf_spatial.json)。评测来源和合并过程见 [评测报告](docs/evaluation_20260926.md)。

**SmolVLA（官方 Spatial 协议）**：论文架构、256×256 官方数据、8D 末端状态、100k 更新、seed 0。把「ramekin 上面」换成 MuJoCo 3.3.2 复测（47/50，该任务原先 10/50）后，全套是 **448/500（89.6%）**。训练中的周期评测是每步重规划，成绩在 68%–73%（100 回合），与上表不是同一个协议。配方见 [smolvla_official.json](configs/libero_spatial/smolvla_official.json)，说明见 [smolvla_official_spatial.md](docs/smolvla_official_spatial.md)。

**Pi-0 LoRA（HF Spatial）**：30k 更新，seed 0，MuJoCo 3.3.2 上每任务 50 回合。闭环成功率是 **438/500（87.6%）**。下图是训练损失。配方见 [pi0_lora_hf_spatial.json](configs/libero_spatial/pi0_lora_hf_spatial.json)。旧的 HDF5 [Pi-0 说明](docs/pi0_lora_spatial.md) 不是上表中的成绩。

![VLA 训练记录](docs/assets/vla_training.png)

## SmolVLA + RL Token 后训练

RLT 从官方 100k SmolVLA Spatial checkpoint 开始，冻结 VLA 和训练得到的 960 维 RL Token，再训练 Gaussian Actor 与双 Q Critic。进度奖励版本通过势函数提供接近目标、有效抓取和搬运进度信号；特权接触与几何信息只用于计算奖励，不输入策略。RLT 和 SFT 对照每次均执行 10 步。

| LIBERO-Spatial 任务（HF 原生顺序） | SFT，C=10 | 稀疏奖励 RLT，C=10 | 进度奖励 RLT，C=10 |
|---|---:|---:|---:|
| 盘子和 ramekin 之间 | 48/50（96%） | 46/50（92%） | 48/50（96%） |
| ramekin 旁边 | 46/50（92%） | 49/50（98%） | 47/50（94%） |
| 桌面中央 | 49/50（98%） | 47/50（94%） | 49/50（98%） |
| 饼干盒上面 | 44/50（88%） | 47/50（94%） | 49/50（98%） |
| 木柜顶层抽屉 | 41/50（82%） | 42/50（84%） | 45/50（90%） |
| ramekin 上面 | 42/50（84%） | 43/50（86%） | 44/50（88%） |
| 饼干盒旁边 | 50/50（100%） | 45/50（90%） | 50/50（100%） |
| 炉子上面 | 41/50（82%） | 46/50（92%） | 45/50（90%） |
| 盘子旁边 | 39/50（78%） | 34/50（68%） | 46/50（92%） |
| 木柜上面 | 42/50（84%） | 38/50（76%） | 39/50（78%） |
| **全套** | **442/500（88.4%）** | **437/500（87.4%）** | **462/500（92.4%）** |

在这次单训练 seed 的评测中，进度奖励 RLT 比匹配的 SFT 对照高 **4.0 个百分点**，比稀疏奖励 RLT 高 **5.0 个百分点**。训练共采集 100,044 个环境步、774 个回合，并进行了 237,155 次 learner 更新；这些 rollout 统计不等同于上表的 500 回合正式评测。三组每任务均评测 50 个固定初态。task 4（木柜顶层抽屉）在修正 hf-libero reset 协议后对三组各重测 50 回合，并核对了物理初态指纹；其他任务复用已存档评测。结果仅代表单 seed，不能说明跨 seed 稳定性。[逐任务结果、协议与 checkpoint 指纹](results/rlt_hf_spatial/summary.json) · [进度奖励配置](configs/libero_spatial/smolvla_rlt_hf_progress.json) · [HF 权重与环境对齐](docs/smolvla_rlt_hf.md) · [奖励定义和验证](docs/smolvla_rlt_progress_reward.md)。

RLT 默认使用官方 HF Spatial 权重和 processor、256px RGB、8D EEF 状态、hf-libero 0.1.4 与 MuJoCo 3.3.2。旧 HDF5 SmolVLA 配方仍可用于复现历史路径（[说明](docs/smolvla_spatial.md)）。RLT 支持按阶段运行：`--stage token`、`--stage warmup`、`--stage online`、`--stage evaluate`，详见[分阶段命令](docs/smolvla_rlt_spatial.md#按阶段执行)。

## 历史 ACT/DP 分析（复测合并前）

原先匹配的 ACT/DP 研究是：ACT 415/500（83.0%），DP final 409/500（81.8%），DP 验证损失最低的 7k checkpoint 347/500（69.4%）。

两组策略共享数据划分、物理动作与评测协议；视觉预训练、裁剪、观察历史、归一化和训练预算不同。因此这是策略方案比较，不是仅架构变化的单因素实验。

**研究发现：离线去噪指标最优，不等于闭环成功率最高。** 同次训练的 final 相对验证最优 checkpoint 提升 12.4 个百分点。DP final 与 ACT 只差 1.2 个百分点；不能凭一个训练 seed 宣称稳定优劣。后验追加的 checkpoint 对比属于探索性分析。

## 项目的实质内容

- 12 个 ACT 模型：K=1/8/16/32 × 3 seeds；每个 checkpoint 对比 chunk、replan 和 TE。
- 一个 suite 多任务 DP；独立 GN CNN、随机初始化、随机/中心裁剪、min–max 动作归一化、EMA。
- 固定 initial state、seed、控制器、相机方向、rollout horizon 和 success predicate；逐 episode 记录便于配对分析。
- 两张 4090 的 CUDA/EGL 显式绑定；环境并行、策略同步；支持图像缓存、预取、断点恢复与源码快照。
- 发布 3,500 条小型评测记录、checkpoint 指纹及绘图程序；无需 GPU 即可复现图表。

## 立即重绘

```bash
python -m pip install -e '.[report,test]'
python scripts/export_official_results.py
python -m roboscope report --study all
python -m pytest tests/test_results.py tests/test_native_results.py tests/test_followup_results.py tests/test_official_results.py tests/test_cli.py
```

正式结果表由 [逐任务 CSV](results/official_spatial/per_task.csv) 和可审计的逐回合来源生成；旧 ACT/DP 研究图与 SmolVLA 历史图仍保留各自的历史口径。

正式训练和评测使用 [quickstart](docs/quickstart.md)。CLI 默认只预览，显式添加 `--start` 才运行。

## 开发方向

代码已分为 `data / policies / envs / trainers / evaluation / runtime / workflows / reporting`，参考 verl-vla 的职责分离方式。Pi-0 LoRA 已通过对应模块接入，可先运行 `bash scripts/train_pi0_lora_spatial.sh --preview` 查看配置。

## SmolVLA + RLinf PPO 后训练

[RLinf `codex/smolvla-ppo` 分支的 SmolVLA 适配](https://github.com/YonderL/RLinf/tree/codex/smolvla-ppo)，从同一份官方 Spatial SFT 100k 权重出发完成了 100 轮 PPO 采样与更新。这是独立于上方四模型正式表和 RLT 的实验。完整 LIBERO-Spatial BF16 评测每任务使用 50 个固定初态，共 500 回合；MuJoCo 3.3.2、ODE10 采样，每次预测 50 步并执行 10 步。PPO100 成功 **449/500（89.8%）**，历史 SFT 对照成功 **442/500（88.4%）**，多 **7 回合（+1.4 个百分点）**；进度奖励 RLT 参照为 **462/500（92.4%）**。木柜顶层抽屉 task4 上，PPO100 为 **43/50**，历史 SFT 为 **41/50**，相同 BF16 模型运行时的 SFT 复测为 **42/50**。这是一组训练 seed 的描述性结果，提升幅度仍需多 seed 验证。

训练期间的 50 回合评测最终为 **40/50**，与该协议下 SFT 基线相同；它与上面的 500 回合固定初态评测不是同一协议。PPO 对每个动作块采样一个 Gaussian flow 转移参与优化；最终评测则在 Gaussian 初始 latent 后使用 ODE 求解。参数、训练细节、运行时限制和复现说明见 [PPO 实验记录](docs/smolvla_ppo_spatial.md)，逐回合轻量记录见 [结果目录](results/smolvla_ppo_spatial/summary.json)，Spatial 的 RLinf 覆盖配置见 [配置文件](configs/rlinf/smolvla_ppo_spatial.yaml)。仓库不包含模型权重、原始轨迹或 benchmark 数据。

**RoboScope 仓库本身尚未实现 PPO trainer、RTC 或跨 benchmark 训练。** PPO 适配代码位于 RLinf；其余方向见 [路线图](docs/roadmap.md)。

原实验目录与权重在本地保留；对外源码包只包含规范代码、配置、文档和轻量结果。重构后的必要验证与尚未完成的整套重跑分别记录在 [validation](docs/validation.md)。
