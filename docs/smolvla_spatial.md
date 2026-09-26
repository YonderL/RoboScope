# SmolVLA on LIBERO-Spatial

新增语言条件 SmolVLA 训练与评测，配置为 `configs/libero_spatial/smolvla.json`。
评测直接复用 ACT/DP 的 `evaluation.worker` 和 `EnvPool`。
按本项目约定，训练和评测命令默认只预览，`--start` 才实际执行。

## 训练设置

从官方 [lerobot/smolvla_base](https://huggingface.co/lerobot/smolvla_base) 微调，
采用 [官方微调示例](https://huggingface.co/docs/lerobot/v0.6.1/smolvla) 的有效 batch size 64、20,000 次更新。
这是官方微调示例的预算，不是从头预训练预算或论文 LIBERO 全套复现。
模型架构、冻结范围、预处理、损失、优化器与调度直接读取官方 base 配置并调用 LeRobot 0.6.1 实现。

| 设置 | 默认值 |
|---|---|
| 初始化 | smolvla_base；缺失或不兼容权重直接报错 |
| 更新次数 / 有效 batch | 20,000 / 64 |
| 显存分批 | microbatch 64；调小后累计到有效 batch 64，按有效动作数加权保留原生 masked loss |
| 训练参数 | 冻结视觉编码器和 VLM，训练 action expert、state/action projections |
| 优化器 | AdamW，lr=1e-4，betas=(0.9,0.95)，eps=1e-8，weight decay=1e-10 |
| 梯度裁剪 | 10 |
| 调度预设 | warmup=1,000，decay=30,000，最终 lr=2.5e-6 |
| 调度实际行为 | 原生 0.6.1 scheduler 在 20,000 步预算下自动缩放 warmup 为 666、decay 为 20,000 |
| 训练精度 | 官方 base 的 `use_amp=false`；原生冻结 VLM 为 BF16、expert/projections 为 FP32；recipe 的 `amp=true` 仅用于共享评测 |
| 观测 / 动作块 / 每次执行 | 1 / 50 / **10** |
| Flow sampling | 原生 10 步 |
| 图像预处理 | 原始 128×128 RGB，经原生 resize-with-padding 到 512×512 |
| 语言 | HDF5 的 `problem_info.language_instruction`，原生换行、tokenizer、长度 48 |
| State / action | 9D joint+gripper / 原始 7D OSC_POSE；按训练集 mean/std 归一化、内部补到 32D |

继续使用项目的按轨迹 90%/10% 划分（split seed 2026），统计量只拟合训练轨迹。
动作窗口从当前时刻开始，末尾重复终止动作并传入 `action_is_pad`，由原生损失屏蔽；不跨轨迹。
原始 OpenGL 图像上下翻转到项目约定的 OpenCV 方向；RGB 通道不交换。
不添加 LoRA、EMA 或 Aloha 动作变换。

训练默认使用最后一张 RTX 4090；SmolVLA 评测默认在该卡上顺序完成奇偶任务分片。
本项目的 GPU/EGL 调度要求机器有两张 RTX 4090。若训练显存不足，可减小 `micro_batch_size`，
保持 `batch_size=64`；累计梯度按有效动作数加权，优化器步数与原生设置一致。

评测可显式传入 `--gpu-model 5880`，使用两张 RTX 5880 并行执行两个分片；
每张卡仍为 8 个环境，总计 500 回合，固定初态、随机种子与执行长度不变。
该选项只影响此次评测，训练的默认 GPU 设置不变。GPU 型号会保存在评测配置中；
不同型号上的推理延迟应分开报告。下面命令评测 SFT 的验证 loss 最优 checkpoint：

```bash
python -m roboscope evaluate \
  --source outputs/smolvla_spatial_seed0 \
  --output outputs/smolvla_spatial_seed0/evaluation_best_5880 \
  --checkpoint best --episodes 50 --ta 10 --gpu-model 5880 --start
```

## 评测协议

| 项目 | ACT/DP 正式对比 | SmolVLA |
|---|---|---|
| 任务 / 回合 | 10 × 50 = 500 | 相同 |
| 初态 | 每任务官方固定初态 0–49，不循环复用 | 相同 |
| 回合种子 | `10000 + 1000*task_id + initial_state_id` | 相同，flow noise 也使用各回合独立 generator |
| 最大策略步数 / 静置步数 | 600 / 5（全零动作） | 相同 |
| 控制与成功判定 | 20 Hz、OSC_POSE、动作裁剪到 [-1,1]、`check_success()` | 相同 |
| 摄像头 | agentview + wrist，128×128 RGB | 相同 |
| 仿真并行度 | 每 GPU 8 个环境，两 GPU 按 task ID 奇偶分片 | 相同 |
| 每次预测执行 | 8 步 | **保留 SmolVLA 原生 50 步**；成功或到达 600 步时提前结束 |
| 视频 | 每任务 initial state 0，20 fps | 相同 |
| 延迟测量 | batch=1，每卡 5 次热身 + 30 次测量 | 相同 |

延迟包括预处理、完整动作块预测和动作 D2H，输入观测已经在 GPU，不含仿真、视频或输入 H2D。
SmolVLA 的语言处理也包含在计时中；完整动作块为 50 步，不能将其耗时直接解释为 8 步动作块耗时。
执行长度与训练配方的差异是本次比较的一部分，不是纯架构消融。
历史 ACT 的 20 回合/任务曲线协议不用于这里的正式评测。

## 运行

先按 [quickstart](quickstart.md) 安装基础训练依赖、LIBERO 资源与原始 HDF5，激活对应环境后：

```bash
python -m pip install -r requirements-smolvla.txt
python -m pip install -e . --no-deps

export DATA_ROOT=/path/to/parent-containing-libero_spatial
export LIBERO_ROOT=/path/to/LIBERO/libero/libero

bash scripts/train_smolvla_spatial.sh --preview
bash scripts/train_smolvla_spatial.sh --start
# 训练完成后自动评测 final checkpoint，500 回合。
# 中断后使用相同配置和输出目录：
bash scripts/train_smolvla_spatial.sh --start --resume
```

脚本支持 `PYTHON`、`OUTPUT_ROOT`、`RECIPE`、`SMOLVLA_BASE_PATH`、`SMOLVLA_VLM_PATH` 环境变量，
以及 `--output DIR`、`--skip-eval`。两个模型路径变量用于已有本地资源。
在线资源在启动前下载到项目 `.cache/smolvla/hub` 并记录解析后的不可变 Hub revision；
可以通过 `HF_ENDPOINT` 配置镜像。不会自动安装依赖或上传模型。

也可使用标准入口：

```bash
python -m roboscope train --recipe configs/libero_spatial/smolvla.json \
  --data-root "$DATA_ROOT" --libero-root "$LIBERO_ROOT" \
  --output outputs/smolvla_spatial_seed0 --start

python -m roboscope evaluate --source outputs/smolvla_spatial_seed0 \
  --output outputs/smolvla_spatial_seed0/evaluation_final \
  --checkpoint final --episodes 50 --start
```

不传 `--ta` 时，SmolVLA 自动使用 50，ACT/DP 仍使用 8。
SmolVLA 不接受 `--ta 8` 或 DP 的 DDIM 步数覆盖。
`--checkpoint best` 可单独评测最小验证 flow loss 权重；输出需换目录。
best 不代表最高闭环成功率，正式默认使用 final。

首次部署可运行 `bash scripts/train_smolvla_spatial.sh --start --smoke`：
独立 `_smoke` 输出目录、2 次训练更新、每任务 1 回合。
它保留 50 步执行长度和 600 步上限，但不构成正式 500 回合评测。

## 产物与验证范围

训练输出包括 `config.json`、`manifest.json`、源码快照、依赖清单、`backend_config/config.json`、
`model_summary.json`、`train_history.json`、`best.pt`、`last.pt`、`final.pt`。
`last.pt` 包含优化器、原生 scheduler 和 RNG 状态，恢复时校验配置、数据清单及源码。
checkpoint 保存完整模型，所需磁盘空间大于仅保存 adapter 的 Pi-0 路线。

评测输出 `summary.json`、`eval/smolvla_chunk10/shard{0,1}/` 下的逐回合 JSONL、轨迹、视频、
延迟原始样本及完成标记，沿用 ACT/DP 的 checkpoint hash 和初态身份核验。

CPU 回归覆盖：预览与配置、原生微型 SmolVLM/expert 前向反向及处理器、严格权重加载与恢复、
数据方向/语言/padding、有效动作梯度加权、优化器断点续训、50 步队列与独立回合 RNG。
另外已在按项目 UUID/EGL 映射选定的 RTX 4090 上运行原生微型模型的前向、反向、
checkpoint 恢复及 BF16 推理测试。主机 CUDA 正常；受限沙箱内的 CUDA 不可见不代表主机故障。
尚未运行完整官方 450M checkpoint 的 GPU 训练或 LIBERO 500 回合评测；此处不报告成功率。

本次新增和原有测试合计 48 项通过，其中原有 CPU 双进程测试因沙箱禁止回环通信，
在主机禁用 GPU 后单独补跑通过。Ruff lint 通过，修改文件格式检查通过。
全仓库格式检查仍有既有 Pi-0 文件不符合格式；源码打包被既有
`docs/pi0_lora_spatial.md` 的个人绝对路径拦截。这两项既有问题未在此次 SmolVLA 改动中调整。
