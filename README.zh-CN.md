# RoboScope

**以可复现的闭环评测为核心，研究视觉机器人策略。**

[English](README.md) · [环境与复现命令](docs/quickstart.md) · [架构](docs/architecture.md) · [实验配置](docs/experiments.md) · [研究分析](docs/findings.md)

当前实现：LIBERO-Spatial 上的多任务 ACT、Diffusion Policy，以及 action chunking、temporal ensemble、DDIM 步数、执行 horizon 和 checkpoint 选择分析。一个模型覆盖十个任务，条件是 task ID，不是自然语言。

![ACT与DP比较](docs/assets/act_vs_dp.png)

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
python -m roboscope report
python -m pytest tests/test_results.py tests/test_cli.py
```

正式训练和评测使用 [quickstart](docs/quickstart.md)。CLI 默认只预览，显式添加 `--start` 才运行。

## 开发方向

代码已分为 `data / policies / envs / trainers / evaluation / runtime / workflows / reporting`，参考 verl-vla 的职责分离方式。后续 π 系列、RL 后训练、其他仿真器通过对应模块接入。

**这些扩展尚未实现，当前不宣称支持 VLA fine-tuning、PPO、RTC 或跨 benchmark 训练。** 优先完成可验证的 ACT/DP 研究，再扩展新能力。[路线图](docs/roadmap.md)

原实验目录与权重在本地保留；对外源码包只包含规范代码、配置、文档和轻量结果。重构后的必要验证与尚未完成的整套重跑分别记录在 [validation](docs/validation.md)。
