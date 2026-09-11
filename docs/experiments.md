# Experiment specification

## Shared protocol

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
