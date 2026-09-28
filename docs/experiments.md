# Experiment specification

## Official result selection

The [README comparison](../README.md#official-closed-loop-results-2026-09-28)
aligns all four policies by task name, with 50 trials per task. ACT, DP and
SmolVLA use their MuJoCo 3.3.2 on-ramekin reruns and the archived other nine tasks;
Pi-0 LoRA uses its complete MuJoCo 3.3.2 evaluation. Official totals are
ACT **434/500 (86.8%)**, DP **444/500 (88.8%)**, SmolVLA **448/500 (89.6%)**,
and Pi-0 LoRA **438/500 (87.6%)**. The [official records](../results/official_spatial/summary.json)
retain each task's source and checkpoint fingerprint. See the
[merge accounting and rerun protocols](evaluation_20260926.md).

## Historical ACT/DP shared protocol

LIBERO-Spatial, 10 tasks, one task-ID-conditioned model per training run. Original 128×128 agentview + wrist RGB. Proprioception: joint positions (7) + gripper qpos (2). Action: OSC_POSE delta translation (3), delta axis-angle rotation (3), gripper (1). Controller at 20 Hz, action clipped to [-1,1]; controller scales translation components to ±0.05 m and rotation components to ±0.5 rad. Gripper uses command sign, not a target width.

45 train / 5 validation demonstrations per task, split seed 2026. Statistics are fitted on training episodes only. Canonical image convention: OpenCV; OpenGL demonstrations are vertically flipped. Official fixed initial states, 5 settling steps, at most 600 policy-controlled steps, success from `check_success()` rather than `done`. Seed = `10000 + 1000*task_id + initial_state_id`. Task IDs follow sorted HDF5 filenames, with BDDL/init matching by name.

## Model and optimization matrix

| Setting | ACT | DP |
|---|---|---|
| Visual backbone | Shared ResNet18, ImageNet initialization, FrozenBN | Independent ResNet18 per camera, random initialization, GN |
| Crop | None | Train random / eval center, 128→116 |
| Visual representation | Spatial feature tokens | SpatialSoftmax, 32 keypoints/camera |
| Core | CVAE + Transformer | Conditional 1D U-Net / FiLM scale+bias |
| Dimensions | Model 512; FFN 3200; encoder4 / decoder1 / VAE4 | U-Net 512/1024/2048; timestep128; kernel5; GN8 |
| Task condition | Learned 512D token | Learned 64D global-condition embedding |
| Observation / prediction horizon | 1 / K=1,8,16,32 | 2 / 16 |
| Action normalization | Train mean/std | Train min–max to [-1,1] |
| Proprioception normalization | Train mean/std | Train mean/std |
| Updates / batch | 56,064 / 32 per model | 30,000 / 128 for the whole suite |
| Seeds | 0/1/2 ablation; 0 matched comparison | 0 |
| Optimizer | AdamW, lr1e-5, decay1e-4 | AdamW, lr1e-4, betas .95/.999, decay1e-6 |
| LR schedule | Constant | 500-step warmup, cosine |
| Objective | Masked L1 + 10×KL | Epsilon MSE, including replicated boundary targets |
| Gradient clip / precision | 10 / BF16 | 1 / BF16 |
| EMA | None | max .9999, power .75 |
| Checkpoints | Last; historical intermediate weights not all retained | Minimum fixed validation MSE / final EMA / resumable last |

DP uses 100 training diffusion timesteps, squared-cosine beta schedule, epsilon prediction. DDIM eta=0, clipping enabled, leading timestep spacing. Its target window is `[a[t-1],...,a[t+14]]`; prediction index1 is `a[t]`. Returned future window has 15 actions; Ta chooses how many to execute. State history updates every environment step.

## Studies and exact denominators

| Study | Training | Evaluation | Checkpoints |
|---|---|---|---|
| ACT chunk / replan / TE | 4 K × 3 seeds | Epoch8/16/24/32; 20 init/task/mode | Corresponding epoch; K1 equivalent modes reused |
| ACT–DP comparison | ACT seed0 K8; DP seed0 | **50 init/task**, 500/model | ACT epoch32; DP 7k and 30k |
| DP DDIM steps | No retraining | 5/10/20, Ta8; 500/config | **7k only** |
| DP Ta | No retraining | 1/4/8, DDIM10; 500/config | **7k only** |
| ACT gripper TE | No retraining | 20 init/task, 200/group | K8 seed1 epoch32; exploratory |

ACT chunk executes K steps per prediction; replan executes the newest first action; TE predicts every step and aggregates predictions for the current target time. TE coefficient .01 favors older predictions. K1 mode records are mathematically reused, not independent rollouts. Historical ACT study: 24,000 physical rollouts. The portable matched/DP dataset contains 3,500 physical rollouts, including the subsequent final test.

## Checkpoint selection and runtime

DP validation: every 1,000 updates, 1,024 fixed validation anchors, fixed noise. Best noise MSE occurred at7k; final30k achieved higher SR. The recipe retains both labels to make the discrepancy reproducible. No final-checkpoint inference sweep has yet been reported.

Each evaluation worker holds one policy; 8 simulator processes per GPU. DP evaluation splits even/odd task IDs between two 4090s. Latency: batch1, five warmups and 30 measurements per card; public plots pool the 60 raw timing samples and show their median. Different checkpoint timings were measured in different sessions, so small differences need not be model effects.

## Native VLA study

Images, proprioception, action execution, rollout budget and episode identity differ from ACT/DP. Figures are rebuilt from [results/vla_spatial/snapshot.json](../results/vla_spatial/snapshot.json) with `python -m roboscope report --study vla`.

| Run | Recipe | What was measured |
|---|---|---|
| SmolVLA, 100k, execute 10 | [smolvla_official.json](../configs/libero_spatial/smolvla_official.json) | **411/500 (82.2%)** |
| SmolVLA periodic rollouts | same training run, execution horizon 1, 100 episodes | 68% at 20k, 73% at 60k, 72% at 80k, 69% at 100k |
| Pi-0 LoRA, HF Spatial, 30k | [pi0_lora_hf_spatial.json](../configs/libero_spatial/pi0_lora_hf_spatial.json) | Training/held-out loss; final closed-loop **438/500 (87.6%)** in `results/mujoco332/` |

SmolVLA here is the paper-architecture run: official 256×256 Spatial data, 8D end-effector state, frozen SmolVLM2-500M, 100k updates, batch 64, seed 0. The 500-episode number executes 10 of the 50 predicted actions. It is not the in-training periodic evaluation, which replans every step. Native reports do not record the ACT/DP initial-state IDs, so the two studies cannot be paired episode by episode. One training seed.

MuJoCo 3.3.2 rerun records and the official merge are documented in [the evaluation report](evaluation_20260926.md). Smoke runs and superseded diagnostic probes remain local.

The earlier [HDF5 SmolVLA recipe](smolvla_spatial.md) (128×128, 9D joints, `smolvla_base`, 20k updates) and the [RLT recipe](smolvla_rlt_spatial.md) remain separate code paths. RLT smoke tests pass; a success-rate comparison has not been measured. The [HDF5 Pi-0 recipe](pi0_lora_spatial.md) is likewise separate from the HF Spatial training curve above.
