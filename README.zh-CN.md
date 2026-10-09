# RoboScope

**面向 LIBERO-Spatial 的机器人策略训练与闭环评测研究平台，让结果能够复现、核对和解释。**

[English](README.md) · [快速开始](docs/quickstart.md) · [实验协议](docs/experiments.md) · [结果与逐回合记录](results/README.md) · [研究分析](docs/findings.md)

我构建 RoboScope，是为了回答一个实际的研究问题：**机器人策略成功率变化时，影响来自模型、checkpoint、动作执行方式，还是评测协议？** 项目把示教数据、策略训练、LIBERO 闭环仿真与可审计的实验结果连接起来，实现了 ACT、Diffusion Policy、SmolVLA、Pi-0 LoRA，以及 RL Token 后训练。SmolVLA+PPO 适配代码位于[我的 RLinf fork 的 `codex/smolvla-ppo` 分支](https://github.com/YonderL/RLinf/tree/codex/smolvla-ppo)。

**4 类策略模型 · 2 条强化学习后训练路线 · 10 项任务 × 每项 50 回合 · 3,500 条已审计 ACT/DP 回合记录**

![ACT、Diffusion Policy、SmolVLA 和 Pi-0 LoRA 在 LIBERO-Spatial 十项任务上的闭环结果](docs/assets/policy_success_rates.png)

## 我完成的工作

| 方向 | 具体贡献 | 代码与证据 |
|---|---|---|
| **训练流水线** | 实现多任务 ACT、Conditional 1D U-Net Diffusion Policy、官方数据 SmolVLA SFT 和 Pi-0 LoRA 工作流，支持 checkpoint 保存与恢复；ACT 实验覆盖 4 种动作块长度 × 3 个训练 seed。 | [实验配置](configs/libero_spatial/) · [工作流](src/roboscope/workflows/) |
| **数据与运行时** | 轨迹级训练/验证划分、仅用训练集统计量归一化、图像方向校验、时序窗口与缓存加载、GPU/EGL 绑定和实验源码快照。 | [数据模块](src/roboscope/data/) · [运行时](src/roboscope/runtime/) |
| **闭环评测** | 固定初态、逐回合独立动作/历史状态、chunk 执行、逐步重规划和 ACT temporal ensemble；明确任务映射、权重身份和仿真协议。 | [评测模块](src/roboscope/evaluation/) · [复现约定](docs/reproducibility.md) |
| **结果审计与分析** | 审计 3,500 条可发布的 ACT/DP 回合记录，分析 checkpoint 与推理参数，并提供无需模型权重或仿真器即可在 CPU 重生成的图表。 | [逐回合记录](results/libero_spatial/episodes.csv) · [研究发现](docs/findings.md) |
| **强化学习后训练** | 在 RoboScope 实现冻结 VLA 的 RL Token 路线，分别研究稀疏奖励和进度奖励；在 RLinf 接入 SmolVLA PPO 并单独完成评测。 | [RLT 方法](docs/smolvla_rlt_hf.md) · [我的 RLinf fork 中的 PPO 代码](https://github.com/YonderL/RLinf/tree/codex/smolvla-ppo) |

ACT 等模型架构和 LIBERO 基准来自已有研究；本项目的工作重点是训练与评测集成、实验控制和可核对的证据。[架构与模块边界](docs/architecture.md)。

## 闭环结果

以下四类基线均完成 LIBERO-Spatial 十项任务评测，每任务 50 回合。表中是仓库**正式选定结果**：ACT、DP、SmolVLA 的「ramekin 上」任务纳入 MuJoCo 3.3.2 修正复测。各类模型的观测、动作与回合协议并不完全相同；这张表展示完整工作流，**不作为纯架构优劣排名**。

| 模型与选用 checkpoint | 成功回合 | SR | 配置 |
|---|---:|---:|---|
| ACT · K=8，epoch 32 | 434/500 | **86.8%** | [ACT](configs/libero_spatial/act_baseline.json) |
| Diffusion Policy · DDIM=10，Ta=8，final 30k | 444/500 | **88.8%** | [DP](configs/libero_spatial/diffusion.json) |
| SmolVLA · 官方 Spatial SFT 100k，执行 10/50 步 | 448/500 | **89.6%** | [SmolVLA](configs/libero_spatial/smolvla_official.json) |
| Pi-0 LoRA · HF Spatial 30k，执行 8 步 | 438/500 | **87.6%** | [Pi-0](configs/libero_spatial/pi0_lora_hf_spatial.json) |

[逐任务成功次数、结果来源及权重哈希](results/official_spatial/summary.json) · [复测协议](results/mujoco332/protocol.json) · [结果合并说明](docs/evaluation_20260926.md)。前三类模型沿用九项历史评测，并替换「ramekin 上」复测；Pi-0 使用完整 MuJoCo 3.3.2 评测。所有完整训练均只有一个 seed。

### 改变 checkpoint 选择的一项发现

在**另一组严格匹配初态的 ACT/DP 实验**中，Diffusion Policy 验证集噪声 MSE 最低的 7k checkpoint 成功 **347/500（69.4%）**，final 30k checkpoint 在相同的 500 个任务/初态组合上成功 **409/500（81.8%）**。选择 final 后，93 个回合由失败变成功，31 个回合反向变化；无需重新训练，净提升 **12.4 个百分点**。这说明该次实验中离线去噪误差没有选出更强的闭环控制器。这项后续 checkpoint 分析是探索性的；配对 bootstrap 和适用边界见[完整分析](docs/findings.md)。

![Diffusion Policy 的验证损失与闭环成功率随 checkpoint 变化](docs/assets/checkpoint_selection.png)

## SmolVLA 后训练：两条路线

两条路线均从官方 Spatial SFT 100k 权重出发，但算法与模型运行时不同。下表 SFT 参照是后训练实验中的 **C=10 对照（442/500）**，不是上方正式四模型表的 448/500。

| 路线 | 实现 | LIBERO-Spatial 结果 | 解读 |
|---|---|---:|---|
| SFT 对照，C=10 | [RLT 对照记录](results/rlt_hf_spatial/summary.json) | 442/500 · 88.4% | 后训练结果的参照 |
| RL Token + 进度奖励 | [RoboScope RLT 工作流](docs/smolvla_rlt_spatial.md) | **462/500 · 92.4%** | 比匹配的 SFT 对照高 4.0 个百分点 |
| PPO，第 100 轮 | [我的 RLinf fork：`codex/smolvla-ppo`](https://github.com/YonderL/RLinf/tree/codex/smolvla-ppo) | **449/500 · 89.8%** | 比历史 SFT 高 1.4 个百分点；单训练 seed |

RLT 冻结 SmolVLA，训练 Gaussian actor 与双 Q critic；接近、抓取和搬运信号只参与进度奖励计算。PPO 则通过 RLinf 的 actor-critic 循环更新 SmolVLA 动作 expert。PPO 完成 100 轮采样与更新，最终使用 BF16 进行了 500 回合评测。木柜顶层抽屉任务上，PPO 为 43/50，相同 BF16 模型运行时的 SFT 复测为 42/50。训练期间另一套 50 回合评测中，PPO 和该协议下的 SFT 基线最终都为 40/50；它与完整固定初态评测不是同一协议。[RLT 结果](results/rlt_hf_spatial/summary.json) · [PPO 参数、协议与 500 条回合记录](docs/smolvla_ppo_spatial.md) · [我的 RLinf fork 中的 Spatial 配置](https://github.com/YonderL/RLinf/blob/codex/smolvla-ppo/examples/embodiment/config/libero_spatial_ppo_smolvla.yaml)。

PPO 观察到的 +1.4 个百分点是单 seed 的小幅描述性收益。历史 SFT 和 PPO 使用不同的 LeRobot/Transformers 模型运行时；若要判断提升是否稳定，还需完整的同运行时 SFT 对照和更多训练 seed。PPO 训练代码在上面的 RLinf fork，**不在** RoboScope 的 trainer 包中。

## 先复现已发布的证据

以下命令在 CPU 上即可核对结果并重生成图表，无需 GPU、LIBERO、数据集或模型权重：

```bash
python -m pip install -e '.[report,test]'
python scripts/export_official_results.py
python -m roboscope report --study all
python -m pytest tests/test_results.py tests/test_native_results.py \
  tests/test_followup_results.py tests/test_official_results.py tests/test_cli.py
```

ACT/DP 对照附有[逐回合 CSV](results/libero_spatial/episodes.csv)；后训练分别发布 [RLT](results/rlt_hf_spatial/summary.json) 和 [PPO](results/smolvla_ppo_spatial/summary.json) 记录。训练或仿真评测见[环境与数据准备](docs/quickstart.md)。训练命令默认先预览，添加 `--start` 才执行。仓库不包含模型权重、示教数据、仿真资源、原始视频或轨迹。[复现约定](docs/reproducibility.md) · [验证记录](docs/validation.md)。

## 仓库结构

```text
src/roboscope/   数据、模型、训练器、评测、运行时、报告与工作流
configs/         版本化实验配置和 RLinf Spatial 覆盖配置
results/         逐回合证据、任务汇总和来源信息
scripts/         训练/导出入口与源码发布工具
docs/            实验协议、分析、运行说明与图表
```

项目为不同模型保留清晰的训练路径，只在数据、运行时和结果审计层复用公共机制；RoboScope 仓库本身不宣称具备分布式 PPO 训练能力。[代码架构](docs/architecture.md) · [参与贡献](CONTRIBUTING.md) · [Apache-2.0 许可](LICENSE) · [第三方说明](THIRD_PARTY_NOTICES.md)。

项目基于 [LeRobot](https://github.com/huggingface/lerobot) 和 [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)。使用相关方法或数据时，请引用 ACT、Diffusion Policy、SmolVLA、Pi-0 和 LIBERO 的原始工作。
