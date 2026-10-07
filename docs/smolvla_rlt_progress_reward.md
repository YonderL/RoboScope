# HF LIBERO 抓取进度奖励（实验配方）

配方：`configs/libero_spatial/smolvla_rlt_hf_progress.json`。相对原 HF RLT 配方，
只新增 `reward` 配置；actor/critic 学习率 3e-4、batch 256、BC 系数 0.1、
网络结构、gamma=0.99、执行 10 步和 replay stride=2 均保持原值。
默认旧配方继续使用稀疏奖励。本实现尚未重新训练，不能据此断言成功率提升。

## 进度定义

从当前任务 BDDL 的唯一 `On(target, destination)` 目标提取目标物体，不能把干扰碗当目标。
每个控制步读取 MuJoCo 碰撞接触和物体位置，仅用于奖励，绝不进入 actor、critic 或 VLM 输入。

```text
reach = 1 - tanh(||target_xyz - eef_xyz|| / 0.10m)
eligible = 目标与至少一侧夹指接触
           且目标没有接触任何非夹指几何体
           且目标中心到 EEF 距离 <= 0.12m
pickup = eligible 连续成立至少 3 次检测，否则为 0
transport = pickup * (1 - tanh(||target_xy - destination_xy|| / 0.20m))
Phi = 0.15 * reach + 0.65 * pickup + 0.20 * transport
```

`Phi` 有界于 [0,1]。`pickup` 是有效抓取的启发式代理，不是完整的力学稳定性判定。
使用整根夹指的碰撞体，而非强制两侧 fingerpad 同时接触，覆盖碗沿抓取。
非夹指接触包括桌面、柜子、ramekin、其他物体和盘子，因此碗仍被支撑或连带底座时
不计抓取进度。该定义不依赖统一桌面高度，适用于从柜子和桌面等不同高度抓取。

连续检测计数属于奖励内部状态，每次 reset 清零，失去条件立即清零；没有保留“曾经抓到”的
最高进度。连续 3 帧确认用于抑制原生接触求解器偶发的一帧断开。正常放到盘子上同样会使
pickup 归零，因此 `pickup_proxy_exits` 不是“掉落失败次数”，需要结合成功/位置/接触解释。
物体若连续多帧与底座失去接触后又回落，仍可能触发该代理；它不能代替完整失败诊断。

## 奖励和 replay

保留原始成功奖励：成功终止步为 1，其余为 0。训练采用 potential-based shaping：

```text
r'_t = r_success_t + 0.1 * (gamma * Phi[t+1] - Phi[t])
R'(t,n) = R_success(t,n) + 0.1 * (gamma**n * Phi[t+n] - Phi[t])
```

第二式严格等于第一式在实际 n 个控制步上的折扣和，适用于跨 action chunk 的 stride-2
窗口及不足 10 步的尾部窗口。观测逐步测量进度，replay 用抵消后的端点式计算；
并非对一个 10 步 chunk 只乘一次 gamma。成功窗口的稀疏部分为 `gamma**(n-1)`。

当前 learner 把成功和 280 步有限时域超时都作为终点，因此两者均令终点 Phi=0，
discount=0。如果以后改成超时 bootstrap，必须同步调整这两项。固定初始状态下，
完整回合的折扣 shaping 和严格等于 `-0.1*Phi[0]`，不会通过反复抓起/掉落刷累计折扣收益，
也不会额外奖励“抓着但超时”。它提供中间 TD 信号，不改变最终成功判据，也不保证神经网络训练效果。

仅持有且进度不变时，单步 shaping 为小负值 `0.1*(gamma-1)*Phi`；不会每步重复发抓取奖金。
下降、正常放置和终止清零可能产生负 shaping，这是折扣进度差定义的一部分，没有另外增加掉落惩罚。

Replay 新增 `sparse_reward`、`shaping_reward`、`potential`、`next_potential`，
learner 仍使用合成后的 `reward`。critic MSE TD loss 和 actor `-Q + BC` loss 不变。

## 日志与运行隔离

逐回合记录 `potential_max`、`pickup_proxy_steps/entries/exits`、`sparse_return`、
`shaping_return` 和 `discounted_shaping_return`。其中 shaping_return 为未折扣诊断值，
不用于优化；验证抵消性质应查看 discounted_shaping_return。
`replay_rewarded_transitions` 继续表示 replay 中合成奖励为正的 transition 数，不能解释为成功样本数。
新增 `replay_success_reward_transitions` 单独统计获得原始成功奖励的窗口。

新配方必须使用独立输出目录、重新采集 warmup/replay。训练已有完整 config 身份校验，
拒绝在原稀疏奖励 checkpoint 上改配置直接 resume；修改 shaping 系数同样不能直接 resume。
旧 replay 没有逐步 privileged geometry，不能未经重新回放标注就混入新奖励训练。
本次没有新增 token 权重迁移入口；旧 token 文件携带旧 config，直接复制也会被身份校验拒绝。
复用已训练 token 应在后续准备新运行时单独校验来源与特征/网络契约，不能绕过校验或混用旧 actor/replay。

固定初态正式评测仍采用原始 benchmark 成功判据，不计算 shaping。已发现的 task 4
HF LIBERO hard-reset 历史问题属于独立的评测协议问题，本次奖励修改不修复它；
新的 SR 对照前仍需统一重置协议。奖励诊断回放复原历史 reset 次数，只用于验证检测信号，
不能充当新的正式评测成绩。

## 验证

`tests/test_rlt_rewards.py` 覆盖进度定义、错抓干扰物、共同支撑、接触抖动过滤、掉落回退、
逐步/重叠 chunk 折扣等价、短窗口、成功与超时终点、缺失数据拒绝、actor 输入隔离和旧奖励 resume 拒绝。
原 RLT / HF / CLI 合约测试同时运行。

可选真实环境测试使用 `ROBOSCOPE_RLT_HF_REWARD_RUN` 指向含 env_config.json 和 manifest.json 的
已有 HF 评测目录，并要求调用端按现有 GPU UUID/PCI/EGL 协议绑定设备。
它使用简单零动作策略执行短回合，贯通真实 worker、collector、replay 和两次 CPU actor/critic 更新；
不进行模型成功率评测。

2026-09-28 验证：59 项常规测试通过；另行开启的真实环境短流程测试 1 项通过，
两次 actor/critic 更新的 loss 均有限。修改文件通过 Ruff lint/format 和 git diff whitespace 检查。

另外按历史环境状态回放了 13 条已保存的动作轨迹（各任务 1 条成功样例 + 3 条失败样例），
最大 EEF 轨迹误差小于 6e-8 m。10 条成功样例均检测到确认抓取；T2/init0 碗沿抓取后掉落
识别到 6 帧确认抓取，进度从掉落前 0.787 降至掉落后 0.0068；T6/init4 未抓取和
T5/init8 连带 ramekin 的确认抓取均为 0 帧。后者加入三帧确认前有 1 帧接触抖动误报。
这些是选定样例的奖励诊断结果，不是 13 回合随机抽样的 SR，也不是检测器的全面准确率。
回放代码与明细分别保存于 `logs/check_rlt_progress_reward.py` 和
`logs/rlt_progress_reward_check/native_replay.json`。

## 正式评测重置协议（2026-09-28）

训练已完成于 100,044 环境步 / 237,155 次更新。正式评测使用新的
`hf_eval_reset_protocol=property_sampler_clear_v1`：每次固定初态 hard reset 前清空
native object_property_initializers，让模型重建只添加一套初始化采样器，避免 HF LIBERO
0.1.4 的累积重复采样器改变 task 4 柜体位置。训练 reset 和训练快照保持原协议。

真实原生环境测试覆盖 task 0 和 task 4：同一初态在不同 reset 历史与新实例之间，
静置后的物理状态和模型 fixture 位姿摘要完全相同。所有新评测 episode 记录该摘要。
新 RLT 评测全部 500 回合，随后补评 SFT/旧 RLT 的 task 4 各 50 回合；其余任务没有
property initializer，复用历史对照。比较脚本逐一校验 task 4 的 50 对物理初态摘要一致，
再输出修正后的三组每任务 SR 和成功平均步数。新评测放在训练目录的新 evaluation 子目录，
历史训练与评测文件不被改写。
