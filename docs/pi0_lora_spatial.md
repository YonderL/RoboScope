# Pi-0 LoRA on LIBERO-Spatial

本实现接入 RoboScope 的 `data / policies / trainers / workflows / evaluation`，复用现有原始 HDF5、整轨迹划分、图像缓存、GPU UUID/EGL 映射和运行目录契约。使用服务器现有 LeRobot 0.6.1 的 PyTorch Pi-0，无需改动 ACT/DP 环境或引入 OpenPI/JAX。模型组件来自 [LeRobot](https://github.com/huggingface/lerobot)，基座是 [lerobot/pi0_base](https://huggingface.co/lerobot/pi0_base)，不是已在 LIBERO 微调过的检查点。

## 在 kty-ly 一键运行

```bash
cd /path/to/roboscope
bash scripts/train_pi0_lora_spatial.sh
```

脚本默认路径针对原训练服务器；在其他机器上通过 `PYTHON`、`DATA_ROOT` 和 `LIBERO_ROOT` 指定 Python、包含 `libero_spatial/` 的数据目录及仿真资源。自动选择两张 RTX 4090（检查时物理编号为 **2、3**），按 UUID 绑定 CUDA，并验证 EGL 的 PCI 映射。训练是两张卡同步更新同一个模型的 DDP；训练完成释放两卡，再分片评测 10 个任务。

首次运行默认通过 `https://hf-mirror.com` 下载约14 GB（13.04 GiB）基座到项目 `.cache/pi0/hub`，解析并保存不可变的 Hub commit；失败会明确停止。可用 `HF_ENDPOINT=https://huggingface.co` 切回官方站或指定其他可用镜像。镜像此前返回429，2026-09-13重新检查已恢复200并开始下载；完整基座训练仍需等待文件下载完成。

Tokenizer使用 [OpenPI 官方代码指定的公开资源](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/models/tokenizer.py)，从 Google `big_vision/paligemma_tokenizer.model` 下载约4.3 MB，并按官方Pi-0格式编码BOS、任务文本和单独的换行符。此资源已实际下载到 `.cache/pi0/openpi_tokenizer`，SHA256身份写入运行配置。脚本仅在缺失时安装 `sentencepiece==0.2.1`，不会升级现有PyTorch或LeRobot。

已有本地资源时可以指定目录：

```bash
PI0_BASE_PATH=/path/to/pi0_base \
PI0_TOKENIZER_PATH=/path/to/paligemma_tokenizer \
bash scripts/train_pi0_lora_spatial.sh
```

基座目录必须含 `config.json`、完整的 `model.safetensors`；tokenizer目录应含 `paligemma_tokenizer.model`，或者是可由`AutoTokenizer`离线加载的PaliGemma tokenizer。代码对权重严格加载，不会在下载失败时用随机模型开始正式训练。

```bash
# 仅预览，不加载GPU、数据或下载权重
bash scripts/train_pi0_lora_spatial.sh --preview

# 2次真实Pi-0参数更新+每任务1回合；输出目录自动加_smoke
# 需要完整预训练资源，此结果只用于检查管线
bash scripts/train_pi0_lora_spatial.sh --smoke

# 只训练；评测可之后独立运行
bash scripts/train_pi0_lora_spatial.sh --skip-eval

# 同一配置/源码/数据继续训练或恢复未完成的评测
bash scripts/train_pi0_lora_spatial.sh --resume
```

脚本在前台运行，可放在已有的 tmux 会话中。训练日志可用 `tail -f outputs/pi0_lora_spatial_seed0/train.log` 查看。通过 `PYTHON`、`DATA_ROOT`、`LIBERO_ROOT`、`OUTPUT_ROOT`、`RECIPE` 环境变量改变运行环境；`--output DIR` 选择输出目录。自定义 recipe 必须另存为 JSON，不能在同一输出目录混用实验配置。

## 微调方案

| 项目 | 默认值及理由 |
|---|---|
| 基座 | Pi-0 / PaliGemma 2B + Gemma 300M 动作专家 |
| LoRA | VLM 与动作专家 attention 的 Q/K/V/O，rank=16、alpha=16、dropout=0 |
| 其余训练参数 | state/action 输入输出投影及 action-time MLP；视觉编码器和其余基座冻结 |
| 精度与显存 | BF16 autocast、梯度检查点、LoRA/机器人投影的 FP32 参数；不复制完整 EMA 模型 |
| 有效 batch | 两卡 × 每卡 microbatch 1 × 累积 16 = 32 |
| 训练预算 | 30,000 次 optimizer update，AdamW，LR 1e-4，warmup 1,000，余弦衰减至 1e-5，梯度裁剪 1 |
| 输入 | 两路 128×128 RGB，Pi-0 内部 resize/pad 到224；真实语言指令；8维末端/夹爪状态 |
| 输出 | 50步动作chunk，32维模型latent，仅真实7维动作参与loss；每次执行前8步，再重新观测 |
| 数据划分 | 每任务50条示教，固定seed=2026，45条train/5条val；统计量只从train估计 |
| 验证 | 每1,000更新，对固定256个验证帧用固定噪声/RNG计算flow-matching loss |
| 保存 | 每1,000更新保存last；保留最低验证loss的best和预定30k的final；保存adapter与机器人投影，无重复14GB基座 |

这是以24GB显存为目标的保守初始配置，实际全模型峰值需下载后运行 `--smoke` 测量。DDP会在每卡各放一份基座，两张24GB卡不能当作单张48GB显存使用。通过 `train_metrics` 记录真实显存和吞吐；不要在未实测前提高 microbatch。

不做额外动作相对化：LIBERO前6维已经是OSC delta命令，最后一维是夹爪命令。末尾不足50步重复最后动作并监督，不跨轨迹。训练图像按照HDF5的`opengl`元信息上下翻转为`opencv`方向；仿真显式输出相同方向的RGB。Pi-0状态为 `ee_states=[xyz,axis-angle] + gripper_states`，与ACT/DP的关节状态适配器分开。语言来自HDF5 `problem_info.language_instruction`，同一指令随manifest进入评测。

## 评测与指标

默认只对预先规定的 **final** 检查点评测：10任务 × 50个固定初始状态，共500回合；每回合10步张开夹爪的settle、最多220个策略控制步；10次flow求解，执行8步chunk。成功以模拟器任务谓词为准，达到步数上限才算失败。程序异常单独报错，不能伪装成任务失败或从分母删去。

220步和settle=10参考 [OpenPI LIBERO evaluator](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py)。本项目使用自己的128像素数据、camera约定和Ta=8，因此不能宣称完整复现OpenPI论文协议。与现有ACT/DP的600步、settle=5历史结果也不能直接当作严格对照。

```bash
# 独立评测final
PYTHONPATH=src python -m roboscope evaluate \
  --source outputs/pi0_lora_spatial_seed0 \
  --output outputs/pi0_lora_spatial_seed0/evaluation_final \
  --checkpoint final --episodes 50 --start

# 可选：探索验证loss最低检查点；与final分开保存，不事后替换主结果
PYTHONPATH=src python -m roboscope evaluate \
  --source outputs/pi0_lora_spatial_seed0 \
  --output outputs/pi0_lora_spatial_seed0/evaluation_best \
  --checkpoint best --episodes 50 --start
```

必要产物：

- `config.json`、`manifest.json`、`gpu_mapping.json`、`requirements.txt`、`source/`：超参数、数据划分/归一化/语言、资源身份、设备映射与源码快照。
- `train_history.json`、`train_metrics.jsonl` / CSV：训练/验证flow loss、LR、梯度范数、样本数、耗时、吞吐、各卡峰值显存。
- `last.pt`：adapter/机器人投影、optimizer、step、各rank RNG及最佳验证值，可恢复；`best.pt`、`final.pt`用于推理。基座路径与版本、数据统计随检查点保留。
- `evaluation_final/summary.json`、`episodes.csv`、`task_metrics.csv`：总成功率、逐任务成功率、Wilson区间、逐回合init/seed/成功/步数/时长、推理耗时。Wilson区间是该固定任务集合的描述性区间，不能代替多训练seed的不确定性。
- 每个分片的 `episodes.jsonl`、`metadata.json`、日志和有限视频：逐回合落盘，恢复时校验checkpoint hash与协议。默认每任务保存首个回合视频，控制磁盘开销。

一次训练seed只能评估管线和该次训练表现。更稳定的研究结果应预先安排至少3个训练seed，独立保存每次训练及评测，再报告均值和离散程度。训练验证loss与闭环成功率分别解释，不以验证loss最佳等同于成功率最佳。

## 验证范围

交付的具体验证结果见本文件末尾的实测记录。CPU/微型模型测试只证明索引、归一化、梯度同步、恢复、指标等代码行为，不能代替真实Pi-0的显存测量或500回合成功率测量。

### 2026-09-12 实测记录

- 在服务器Python3.12.14、PyTorch2.7.1+cu118、Transformers5.5.4、LeRobot0.6.1环境，32项数据/策略/训练/评测/CLI契约测试全部通过。中断后恢复的微型模型参数与未中断训练逐元素完全一致。
- 在两张真实4090上执行NCCL双进程微型LoRA训练，4次更新后的两rank参数完全一致，保存/恢复测试通过；这不是完整Pi-0训练。
- 两张4090均完成真实EGL环境reset、Pi-0输入适配、2个策略控制步及MP4保存。该检查使用明确标注的dummy动作，不产生Pi-0成功率结论。
- 完整数据检查：450条train轨迹、50条val轨迹，分别56,053和6,197帧；两卡物理编号2/3、EGL映射2/3已验证。
- 使用当前真实Pi-0架构在meta设备检查，144个attention投影成功注入LoRA，9,441,312个参数可训练，梯度检查点启用。此检查不分配完整权重或GPU显存。
- 公开tokenizer已下载4,264,023字节并验证编码，资源身份为 `local-sha256:8c43d9e67916183a7aae79bfc054c7298c7a8b1a67e7db3bf5ffb75ddeb0f13f`（文件名及内容的组合摘要）。基座镜像GET/API返回429，下载失败时训练入口按预期停止。
- 未完成项：完整预训练权重严格加载、真实Pi-0反向显存/吞吐测量、30k训练、500回合成功率。下载可用后先运行`--smoke`，确认真实模型训练与推理，再执行正式一键命令。

### 2026-09-13 部署复核

- 已将18个新增/修改文件部署至项目工作目录；原文件保存在本地 `.backups/pi0_lora_20260912T163514Z`。
- 正式项目完整测试集40项全部通过，Ruff检查和启动脚本语法/预览通过。
- 镜像恢复200，当前基座commit为 `25c379b52ba2ff8788cab921758a3cc3fe3f77f2`。下载中的safetensors文件头与已安装Pi-0架构严格匹配，778个张量名称及形状一致；该检查不等同于完整权重加载或前向/反向验证。
- 复核时基座正在下载，tokenizer已缓存。正式训练尚未启动；训练配置仍为每卡microbatch 1、累积16次、双卡有效batch 32。实际显存峰值仍待完整基座短跑测量。
