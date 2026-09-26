# RoboScope

**以可复现的闭环评测为核心，研究视觉机器人策略。**

[English](README.md) · [环境与复现命令](docs/quickstart.md) · [架构](docs/architecture.md) · [实验配置](docs/experiments.md) · [研究分析](docs/findings.md)

当前实现：LIBERO-Spatial 上的多任务 ACT、Diffusion Policy，以及 action chunking、temporal ensemble、DDIM 步数、执行 horizon 和 checkpoint 选择分析。ACT/DP 使用 task ID 条件。另外两条语言条件路线单独记录：官方数据上的 SmolVLA，以及 HF Spatial 上的 Pi-0 LoRA。它们的图像、状态和回合协议与下方 ACT/DP 表不同，不并入该比较。

![ACT与DP比较](docs/assets/act_vs_dp.png)

**SmolVLA（官方 Spatial 协议）**：论文架构、256×256 官方数据、8D 末端状态、100k 更新、seed 0。每次执行 10 步的闭环结果是 **411/500（82.2%）**；同一权重每任务 10 回合为 81/100。训练中的周期评测是每步重规划，成绩在 68%–73%（100 回合），与 82.2% 不是同一个协议。配方见 [smolvla_official.json](configs/libero_spatial/smolvla_official.json)，说明见 [smolvla_official_spatial.md](docs/smolvla_official_spatial.md)。

![SmolVLA 原生评测](docs/assets/smolvla_evaluation.png)

**Pi-0 LoRA（HF Spatial）**：30k 更新，seed 0。发布内容是下图中的训练损失和留出轨迹损失，没有完整闭环分数。配方见 [pi0_lora_hf_spatial.json](configs/libero_spatial/pi0_lora_hf_spatial.json)。旧的 HDF5 [Pi-0 说明](docs/pi0_lora_spatial.md) 是另一条路径。

![VLA 训练记录](docs/assets/vla_training.png)

另有两条尚未报告成功率的路径：走共享评测器的 HDF5 SmolVLA（[smolvla_spatial.md](docs/smolvla_spatial.md)），以及 RLT 后训练。RLT 冻结 SFT，用 RL token、Gaussian actor 和双 critic 在 LIBERO-Spatial rollout 上更新；每次执行 10 步，并保留 SFT 的 10 步对照。可用 `bash scripts/posttrain_smolvla_rlt_spatial.sh --preview` 预览，详见 [RLT 运行说明](docs/smolvla_rlt_spatial.md)。仿真短测试已通过，成功率增益尚未测量。

支持 [分阶段执行](docs/smolvla_rlt_spatial.md#按阶段执行)：SFT → `--stage token` → `--stage warmup` → `--stage online` → `--stage evaluate`。先冻结 RL token 再保存特征 replay；在线阶段持续采集新轨迹并更新策略。

| 配置 | 成功率 | 测试量 |
|---|---:|---:|
| ACT，K=8，epoch32，chunk | **83.0%** | 415/500 |
| DP，final 30k，DDIM10 / Ta8 | **81.8%** | 409/500 |
| DP，验证 noise MSE 最优 7k，同推理配置 | 69.4% | 347/500 |

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
python -m roboscope report --study all
python -m pytest tests/test_results.py tests/test_native_results.py tests/test_cli.py
```

正式训练和评测使用 [quickstart](docs/quickstart.md)。CLI 默认只预览，显式添加 `--start` 才运行。

## 开发方向

代码已分为 `data / policies / envs / trainers / evaluation / runtime / workflows / reporting`，参考 verl-vla 的职责分离方式。Pi-0 LoRA 已通过对应模块接入，可先运行 `bash scripts/train_pi0_lora_spatial.sh --preview` 查看配置。

**PPO、RTC 或跨 benchmark 训练尚未实现。** 新增能力的代码验证与完整训练成绩分别记录。[路线图](docs/roadmap.md)

原实验目录与权重在本地保留；对外源码包只包含规范代码、配置、文档和轻量结果。重构后的必要验证与尚未完成的整套重跑分别记录在 [validation](docs/validation.md)。
