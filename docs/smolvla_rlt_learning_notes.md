# SmolVLA RLT：读代码与训练排查

以下默认值已对齐到 [HF Spatial 协议](smolvla_rlt_hf.md)：256px RGB、8D EEF 状态、280 步回合。

本文针对仓库当前实现，区分代码确定的行为和需要实验验证的风险。
所有调参建议都是待验证方案；本次只补充注释与分析，没有改动算法、默认参数或评测协议。
训练入口和论文适配说明见 [RLT 运行说明](smolvla_rlt_spatial.md)。

## 建议的阅读顺序

1. `trainers/smolvla_rlt.py`：先看 `train_token` 和 `train_online`，理解两个阶段分别训练谁。
2. `policies/smolvla_rlt.py`：看 `describe`，了解图像如何变成状态向量和参考动作。
3. `rl/collector.py` → `rl/replay.py`：跟踪真实执行的动作如何成为一个 TD 样本。
4. `rl/learner.py` → `rl/networks.py`：理解 TD 标签、梯度方向及 Q/actor 的更新顺序。
5. `rl/token.py`：最后查看瓶颈表示如何用因果重建学习。

默认 batch 的关键形状如下。B 是 batch size，N_image 是图像位置的数量（保留相机有效性 mask）。

| 张量 | 形状 | 含义 |
|---|---|---|
| VLM image features | `[B, N_image, 960]` | 冻结 SmolVLA 最终层的图像位置特征（SmolVLM2-500M） |
| RL token | `[B, 960]` | 与 VLM hidden 同宽的单个压缩向量；无 960→更窄投影 |
| state / next_state | `[B, 969]` | token + 8D EEF 状态 + 剩余时间比例 |
| reference / action | `[B, 10, 7]` | SFT 参考动作 / 真实执行的动作 |
| action_mask | `[B, 10]` | 短尾窗口哪些动作实际执行过 |
| reward / discount | `[B]` | 窗口折扣奖励 / bootstrap 系数 |
| twin Q | `[B, 2]` | 两个网络对整段动作的价值估计 |

第一阶段梯度只流入 token encoder/decoder。第二阶段两者均冻结：critic 学习回报，
actor 通过 `action -> Q` 的导数学习动作；冻结 critic 参数不等于切断这个导数。
最终改善的是“冻结 SmolVLA + token + actor”的组合策略，SmolVLA 自身参数没有被 RL 更新。

### 特征取自哪一层？新增 Transformer 有几层？

`prefix_prefill` 取冻结 VLM 主干**最后一个实际保留的 Transformer block 之后，再经过最终
`text_model.norm`（RMSNorm）的输出**。官方 [smolvla_base 配置](https://huggingface.co/lerobot/smolvla_base/blob/main/config.json)
为 `num_vlm_layers=16`，因此对应第 16 层（从零编号的索引 15）之后的最终归一化特征。
实际深度由载入的 SFT 架构决定，不在 RLT 中硬编码；微型测试模型可以只有 2 层。

`outputs[0]` 中的 `0` 指 VLM 分支，不是层号。它先产生完整 prefix 的输出，随后
`image_features_from_prefix` **仅选择图像位置**送进 RL token encoder 和重建目标，
排除语言/state 位置及 prefix 尾部 padding。这些特征经过了 VLM 主干，而非单独视觉编码器的输出。
边界根据原生 attention 标记和包含 padding 的语言序列长度计算，不能用有效 token 总数猜测。

论文图 2 输入重建网络的是 image embeddings，脚注 1 明确实验省略语言 embeddings，
公式 (3) 则将本体状态与 RL token 拼接。当前实现遵循这一设置；原先保留完整 prefix 的版本已更正。
语言依然输入冻结 VLA，并可能通过注意力影响图像位置的输出，但不作为独立 token 被重建。
本体状态在下游单独拼接为归一化的 8D EEF 向量，不使用 VLM 的 state token 作为 RL token 输入。

新增的 RLToken encoder 与 reconstruction decoder **各 2 层**，参数互不共享，
均为宽 960（等于 SmolVLM `hidden_size`）、8 个 attention heads、FFN 宽 3840，
使用 pre-LayerNorm、GELU、零 dropout，末尾再加 LayerNorm。
`token_layers=2` 同时指定两者各自的深度，不是两个网络合计 2 层。
宽与特征维相同时，encoder/decoder 入口不做线性降维，直接使用 stop-gradient 的 VLM embedding；
仅保留论文公式 (2) 的输出投影 `h_φ`。`build_policy` 拒绝 `token_dim != VLM hidden_size`。
decoder 用 `TransformerEncoder` 加上三角 Bool mask（True=禁止 attend）实现因果自注意力，
没有 cross-attention。在线推理只需要 encoder 输出的 RL token；decoder 仅用于第一阶段重建训练。

```text
图像 / 语言 / 状态 prefix
  -> 冻结 VLM 的 16 个保留 blocks -> 最终 RMSNorm -> [B, M, 960]
  -> 仅选图像位置 [B, N_image, 960]，排除语言/state/padding 位置
  -> 直接进入宽 960 的 encoder、追加可学习 readout（无降维投影）
  -> 2 层 encoder -> 取 readout 位置 -> RL token [B, 960]
  -> RL token + 右移后的图像特征 -> 2 层因果 decoder + 输出投影 -> 重建 VLM 图像特征

在线 RL：RL token + 归一化本体状态 + 剩余时间 -> actor/critic
```

## 1. 最先检查：有没有成功奖励

当前奖励只有首次 `check_success()` 为真时的 1；抓到物体、靠近目标等进展没有奖励。
如果 SFT warmup 全部失败，critic 没有观察到正回报，actor 主要受参考动作约束和不准确 Q 梯度驱动。
探索仍可能产生成功，但单纯增加更新次数不会凭空产生成功经验。

先看 `rollouts.json` 的 `success`、`warmup` 和 `replay_rewarded_transitions`，按 `task_id` 分组。
后者是正奖励**窗口数**，不是成功回合数：C=10、stride=2 时一次成功通常贡献最多 5 个正奖励窗口。
一个 280 步的成功回合有 140 条窗口，直接含正奖励的比例最多约 3.6%，其余依赖 bootstrap。

默认 warmup=6000 控制步：如果全是 280 步失败回合，需要 22 个完整回合，即每任务约两次。
默认总预算 100000 步在同样条件下约为 358 个回合，每任务仅约 35–36 个。
这些是根据配置计算的覆盖量，不能把“十万步”理解为十万次独立尝试。

排查顺序：先验证 SFT C=10 基线，再查看随机训练 reset 下各任务能否成功，最后决定是否增加
warmup/采集预算。成功重采样或奖励塑形可能有帮助，但都属于新实验，不能悄悄改变当前算法。

## 2. Actor 接管后可能比 SFT 差

当前 actor 是随机初始化的直接动作网络，不是给 SFT 输出叠加一个零初始化残差。
reference 是输入和正则目标，不保证 actor 初始输出等于它。
warmup 后先进行 `initial_updates=1000` 次额外 critic 更新，以及本回合对应的 UTD 更新；
其中每两次才更新一次 actor。这是联合 Q+正则训练，没有“先纯 BC 学会 SFT”的阶段，也没有接管门槛。

如果 warmup 有成功、`warmup=false` 后成功骤降，先检查 actor 对 reference 的偏差及动作视频。
固定 std=0.05 的逐坐标探索可能影响抓取和夹爪时机，尤其在接触动作中。
未来可以单独验证 BC 初始化、接管前的开发集检查或调整预更新预算；本次没有加入这些策略。
不能仅凭增加预更新次数认定更稳定，因为 critic 也可能在有限数据上过拟合。

## 3. Loss 下降不等于成功率上升

critic 学习的是自己 bootstrap 出来的标签，actor 又会追逐 critic 认为价值高的动作。
未充分覆盖的动作可能得到错误高分。双 Q 取最小值、延迟 actor 更新和 target 平滑旨在缓解这个问题，
不能消除它。这一机制可对照 [TD3 原论文](https://proceedings.mlr.press/v80/fujimoto18a.html)。

如果 `q` 上升、`actor_loss` 下降，但成功率下降，需要怀疑价值外推错误。
当前一次成功就结束且只有一次奖励 1，真实折扣回报应在 [0,1]；持续明显超出这个范围是诊断信号。
网络输出并没有硬裁剪到 [0,1]，初期轻微负值并不直接证明实现错误。

stride=2 的相邻窗口共享 8/10 动作，UTD=5 又对每个新增窗口做 5 次更新。
280 步回合会触发 700 次更新（第一次还有额外预更新），但并没有 700 份独立的新经验。
若出现这种退化，优先在独立开发初态上对比降低 UTD、降低学习率和增加采集多样性，逐项修改。

## 4. gamma 和 beta 可能让优化目标偏离你的预期

`gamma=0.99` 按**控制步**折扣，不是每 10 步才折扣一次。
完整窗口的 bootstrap 系数是 `0.99**10 ≈ 0.904`。
一条从开始到成功共 280 步的轨迹，初始回报约为 `0.99**279 ≈ 0.0606`；
140 步成功约为 `0.99**139 ≈ 0.2473`。因此当前目标明显偏好更早成功，而正式指标只看 280 步内是否成功。
长链任务的早期价值信号可能很弱。

actor 正则是 **70 个坐标的平方和**，再乘 beta=0.1，不是每坐标平均误差。
满 10 步窗口若每坐标 RMS 偏差为 0.1，正则约为 0.07；偏差为 0.3 时约为 0.63。
这可能远大于早期状态的 Q 值，也可能在困难状态不足以限制错误 Q 梯度。
数值量级只供诊断，真正决定更新方向的是两项的梯度。

观察 `bc_weight * bc_penalty` 与 Q 的量级，同时看成功率和视频。若始终复制 SFT、没有收益，
或远离 SFT 后不稳定，分别检查正则过强/过弱的可能。
更高 gamma 可能保留远期成功信号，也可能增加估值困难；需新建实验验证，不修改旧 run 后强行 resume。

## 5. RL token 重建很好，仍可能不适合控制

重建 MSE 不是任务损失；teacher forcing 让 decoder 利用前面的特征预测后面的特征，
因此低 MSE 不足以证明 token 保留了抓取接触、目标位置或经 VLM 融合的语言任务信息。
当前 token 只用 SFT 演示训练，RL 失败后的画面可能超出它的训练分布。
单帧 token + joint/gripper 也不保证能区分所有具有不同速度或接触历史的物理状态。

可在保留的演示/开发 rollout 上查看重建误差，并做 token 置零或置乱的诊断，观察重建和控制是否受影响。
这些消融当前没有自动实现。不要为了提升表示而在 RL 中直接解冻 token：
replay 缓存的是旧 token 向量，解冻后需要额外处理历史表示失效的问题。

## 6. 多任务平均数可能掩盖困难任务失败

任务按回合轮转，replay 却按 transition 均匀采样。
长失败回合贡献更多窗口，短成功回合贡献更少；回合轮转不等于训练样本按任务平衡。
容量满后环形 replay 会覆盖旧样本，成功数据也不会永久保留。
在默认 100000 步、stride=2、capacity=50000 下，通常接近训练末期才填满；
若延长训练或缩小容量，这个问题更突出。

必须同时看每任务成功率、回合长度和 buffer 中各任务的样本占比。
任务均衡采样/成功样本保留是可以研究的改动，但会改变采样分布，目前尚未实现。

## 7. 短尾窗口和时间终止是建模假设

当前成功或达到 280 步都不 bootstrap，并在状态里加入剩余时间，符合这里的有限时域任务定义。
不要照搬“truncated 一定要 bootstrap”或“一定不能 bootstrap”的规则到所有环境。

成功的短尾窗口补零并使用 mask；learner 的 actor loss 也沿用该 replay 样本的 mask。
这意味着评价 actor 替代动作时，沿用了原行为实际结束的窗口长度，而替代动作未必会同样早成功。
这是一项近似，mask 不能自动消除其偏差。未来可在开发集比较尾部窗口比例和其他终止建模方法，
但不能把所有含成功奖励的尾部样本直接删掉，否则会丢失重要信号。

## 8. 采集慢、GPU 利用率起伏，不一定是 CUDA 出错

actor/critic 很小，但每个新 reference 仍需运行 SmolVLA 的完整 50 步 proposal。
280 步回合中，实际控制约查询 28 次；stride-2 replay 需要约 140 个状态的 reference，
约 112 个状态在回合结束后批量补算。这里是状态数，不是 batch 调用数。
相比只在 10 步边界做推理，特征/参考采集开销明显增加。

目前使用单环境同步采集与学习，`workers` 只控制 token 阶段的数据加载，
`eval_envs=8` 只控制评测；修改它们不会让在线采集变为 8 个环境。
一回合 RGB 暂存约 110 MB（280×2×256×256×3 字节，不含 Python 开销），
replay 主要浮点数组容量约 148 MB，另有 mask、元数据和 checkpoint 序列化的临时副本。
每回合保存完整 replay 还会消耗磁盘带宽。

显存不足时先减 `feature_batch_size`，它同时影响 token 特征提取和中间 reference 补算；
RL 的 `batch_size` 不控制 VLA 特征提取批量。当前日志没有细分阶段耗时，需先测量再认定瓶颈。

## 9. 训练成功率不能直接当作评测结果

训练是随机 reset + 带探索噪声的 actor；正式评测是固定初态 + actor 均值。
两者分布和动作采样不同。RLT C=10 应首先与冻结 SFT C=10 对照，另保留原 SFT C=50。
500 回合中一个回合就是 0.2 个百分点，单个任务 50 回合中一次成败是 2 个百分点。
小幅差异需结合固定初态的配对成败、多训练种子和统计不确定性解读，不能只看总平均。

调参使用独立开发初态；不要反复用最终 500 回合挑 beta/gamma/checkpoint 后仍将其当作未见测试集。
当前程序只导出 final，不按正式评测成功率挑 best，但仍需要实验使用者保持开发/测试分离。

## 当前日志能回答什么

| 已有字段 | 可用来判断 | 不能据此断言 |
|---|---|---|
| `success`, `task_id`, `warmup`, `steps` | 各任务采集成功和接管前后变化 | 正式评测成功率 |
| `replay_size`, `replay_rewarded_transitions` | buffer 大小和正奖励窗口是否存在 | 独立成功次数、任务均衡程度 |
| `q`, `target_q`, `critic_loss` | 最后一次更新的估值与拟合误差 | critic 已准确或整回合训练稳定 |
| `actor_loss`, `bc_penalty` | 最后一次 actor 更新的目标量级 | 策略一定比 SFT 好 |
| `reconstruction_loss` | 当前 token 训练 batch 的压缩误差 | 任务信息充分或泛化成功 |

当前每回合只记录最后一次 learner 更新；若最后一次恰好不更新 actor，该条日志没有 actor loss。
尚未记录每次更新均值/分位数、双 Q 差距、动作饱和比例、分任务 replay 占比、梯度范数和分阶段耗时。
这些是后续最有价值的诊断项，但本文没有把它们描述成已经存在的功能。

建议顺序：先确认 SFT C=10 有基础能力 → 检查 warmup 奖励与任务覆盖 → 查看接管后是否退化
→ 检查 Q 与正则量级 → 用独立开发初态做单变量实验 → 最后运行固定 500 回合正式对照。
原方法来源为 [RLT 论文](https://arxiv.org/abs/2604.23073)；以上 LIBERO 风险和默认预算分析是对本仓库实现的推断，
不能理解为论文已经证明这些参数能在 SmolVLA/LIBERO 上带来收益。
