# SmolVLA RLT：HF 权重与环境对齐

2026-09-27 起，后续新实验统一采用 HuggingFaceVLA/libero 数据与 pip 安装的 hf-libero 仿真实现。
RLT 默认从已完成的官方 Spatial SmolVLA SFT 开始；旧 HDF5 配方和历史成绩保留用于复现。
本次改变的是 RLT 的输入和运行协议，不重新训练 ACT、DP 或 Pi-0。

## 固定协议

| 项目 | 新 RLT 与 SFT 对照 |
|---|---|
| 默认 recipe | `configs/libero_spatial/smolvla_rlt_hf.json` |
| SFT 来源 | `outputs/smolvla_official_spatial_seed0`，100k 更新 |
| 权重 | `training/checkpoints/100000/pretrained_model/model.safetensors`，严格加载 |
| 数据 | 本地提取的 `HuggingFaceVLA/libero` Spatial，revision `86958911c0f959db2bbbdb107eb3e17c5f9c798e` |
| Token 演示 | 与官方 SFT 相同的全部演示，52,970 帧；不重新划分、不重算归一化统计 |
| 图像 | 双相机 256×256 RGB；HF 演示保持原方向；原生环境 OpenGL 图像经官方 processor 旋转 180° |
| 状态 | EEF xyz + quaternion 转 axis-angle + 两维 gripper qpos，共 8 维 |
| 归一化 | 加载 checkpoint 的 pre/postprocessor JSON 和 safetensors；包含保存的 epsilon |
| 动作 | 原始 7D 相对 OSC_POSE；预测 50 步，执行 10 步；环境动作裁剪到 [-1,1] |
| RL 状态 | 960D token + 8D normalized EEF + 1D remaining time = 969D |
| 仿真 | hf-libero 0.1.4、MuJoCo 3.3.2、robosuite 1.4.0、LeRobot 0.6.1 |
| Reset | 原生 LeRobot `LiberoEnv`，hard reset，20 Hz，10 步 `[0,0,0,0,0,0,-1]` 静置 |
| 回合上限 | 280 控制步，与当前原生 Spatial 默认上限一致 |
| 初态 | 训练随机 reset；评测各任务固定初态 0–49，不允许循环补足 |
| 默认 GPU | PRO 5000；训练使用其中一张，经 UUID/PCI/EGL 映射绑定 |

HF 数据集的 task index 与 native benchmark 顺序不同。Token 数据按完整语言指令映射到
native task ID；例如 “on the ramekin” 使用 native T5，不使用旧 HDF5 的 T7。

只将资产路径指向本地 `LIBERO/libero/libero` 下的 BDDL、初态和 assets，Python 仿真实现必须来自
安装的 hf-libero。启动时检查版本与模块来源，拒绝被本地旧 checkout 覆盖的 `libero`。

`last` 在准备阶段解析为实际 checkpoint 目录，并核验训练步数已经达到配置的最终步数。
RLT 保存权重、processors、SFT config/manifest、HF metadata、BDDL 和初态文件的指纹；
恢复和加载时重新校验。旧的 9D token、actor、critic 与 replay 不能恢复到新协议。

## 运行

在项目根目录使用 `pi0-blackwell` 环境。脚本依然默认只预览。

```bash
conda activate pi0-blackwell
export PYTHON="$(command -v python)"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

# 验证真实 SFT 权重、HF 演示输入与 native simulator；不进行训练。
python -m roboscope.workflows.smolvla_rlt_check \
  --source outputs/smolvla_official_spatial_seed0 \
  --recipe configs/libero_spatial/smolvla_rlt_hf.json \
  --output outputs/smolvla_rlt_hf_alignment_new

# 预览新的默认来源和配置。
bash scripts/posttrain_smolvla_rlt_spatial.sh --preview

# 限定短流程，输出自动添加 _smoke；不能作为正式结果。
bash scripts/posttrain_smolvla_rlt_spatial.sh --smoke --start --skip-eval

# 正式训练使用全新的默认目录，不恢复 smoke 数据。
bash scripts/posttrain_smolvla_rlt_spatial.sh --stage token --start
bash scripts/posttrain_smolvla_rlt_spatial.sh --stage warmup --start --resume
bash scripts/posttrain_smolvla_rlt_spatial.sh --stage online --start --resume --skip-eval
bash scripts/posttrain_smolvla_rlt_spatial.sh --stage evaluate --start
```

默认正式预算为 10,000 次 token 更新、6,000 步 warmup、共 100,000 控制步（包含 warmup）。
正式运行前可另存 recipe 调整预算；同一 run 的配置和源码必须保持不变。
最终 SFT/RLT 对照各 500 回合，均使用上述统一协议。历史 89.6% 汇总不直接作为新实验的配对对照。

## 对齐验证

真实 100k 权重在 PRO 5000 上通过检查：3 个 HF 数据样本、native T0/T5 固定初态、随机训练初态。
相同噪声下，RLT 的 SFT 参考动作与原生模型前 10 步经裁剪后的最大绝对误差均为 **0**。
原生状态归一化与 RL 使用的 8D 状态也通过一致性检查。
本地报告：`outputs/smolvla_rlt_hf_alignment_20260927_v3/alignment.json`。

完整短流程也已通过：2 次 token 更新、6 个回合 / 120 控制步（其中 1 个 warmup 回合）、
60 条 replay transition、62 次 critic 更新。恢复后没有重复采集；RLT 与冻结 SFT 两个评测入口
分别完成 native T0/T5 各一个 20 步回合。该预算仅用于检查流程，不计算有意义的成功率。
短流程汇总：`outputs/smolvla_rlt_hf_check_20260927_smoke/validation.json`。

48 项 SmolVLA/RLT/CLI 测试、25 项共享环境/评测/数据回归测试，以及仓库要求的
results/CLI 基础测试均通过；全项目 Ruff 检查和格式检查通过。

对齐时发现 CPU 与 CUDA 的直接 `/255` 运算可有一个 FP32 舍入单位的差异，并在 BF16 推理中
放大为动作差异。HF 适配器采用 CPU 构建的 256 项像素映射，在 GPU 上得到与演示逐元素相同的输入。
检查直接比较原生演示预处理、原生动作生成、适配器动作生成和 RLT prefix/cache 复用路径。

这些检查验证接口兼容性，不代表 RLT 已带来成功率提升。正式在线训练和统一 500 回合评测仍需执行。

## Token 注意力数值问题（2026-09-27）

在 PyTorch 2.7.1+cu128、RTX PRO 5000 Blackwell 上，960 维、8 heads 的 FP32 token
训练默认进入 memory-efficient SDPA。一次运行的裁剪前梯度范数由第 820 步的 613
升至第 870 步的 2.7e18，随后在非有限范数检查处退出；最后完整 checkpoint 为第 750 步。
这不是显存不足。已有 `grad_clip=10` 和 Adam `eps=1e-6` 未能阻止失稳。

使用第 750 步权重及第 751 步的同一组真实特征，固定输入进行对照：

| 注意力实现 | 重建 loss | 梯度范数 | 梯度与 FP64 参考的 L2 差 |
| --- | ---: | ---: | ---: |
| Math FP64 参考 | 0.204250967 | 0.285827421 | 0 |
| Math FP32 | 0.204250984 | 0.285827442 | 2.07e-7 |
| 默认 FP32（一次观测） | 0.203412775 | 0.909124819 | 0.848454334 |

Profiler 确认默认路径为 `_scaled_dot_product_efficient_attention`；重复计算还出现了不同结果。
这些对照确认了该运行环境下的注意力数值异常，但没有保存原崩溃瞬间的完整状态，不能声称已逐步重现
原先的失稳轨迹。原实现从第 750 步的独立诊断运行到第 920 步也未再次崩溃。

`RLToken.encode/reconstruct` 现在局部使用 `SDPBackend.MATH`，不改变冻结 VLM 的实现、
token 参数结构、重建 loss、学习率或有效 batch。实际输入的修复后梯度与 FP64 相对误差约 7.2e-7。
生产维度的 CUDA 回归测试修复前失败、修复后通过；运行时需明确选择 GPU 并设置
`ROBOSCOPE_CUDA_TESTS=1`。CPU、CUDA、RLT、HF 相关检查合计 23 项通过。

原运行保留在 `outputs/smolvla_rlt_hf_spatial_seed0_d960_b64_lr3e5_eps1e6_mb16`，其
`diagnostics/` 保存输入、脚本及数值对照。修复后从第 750 步完整迁移权重、优化器和 RNG 至
同名追加 `_math` 的新目录；旧 checkpoint 包含旧实现的训练历史，并非从头用新实现训练。
恢复后已验证至第 1010 步，loss 为 0.16987、裁剪前梯度范数为 0.04438；观察区间日志中的
最大梯度范数为 0.15825。第 1000 步 checkpoint 的参数与优化器状态均为有限值，
实测约 0.506 秒/步。该检查覆盖原崩溃区间，尚不代表完整 10,000 步训练已完成。
