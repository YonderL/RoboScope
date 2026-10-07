# Architecture and extension boundaries

This layout follows the responsibility separation of [verl-vla](https://github.com/verl-project/verl-vla), not its distributed implementation. There is no Ray/verl dependency and no pretend RL backend.

```mermaid
flowchart LR
    C[Scientific recipe + local paths] --> W[Workflow for one recipe family]
    W --> D[HDF5 or HF/LeRobot adapter]
    D --> T[Trainer for that policy]
    T --> K[Checkpoint + optimizer + RNG state]
    K --> E[Evaluation worker]
    E --> P[Policy prediction]
    P --> Q[Per-episode action queue]
    Q --> S[LIBERO environment]
    S --> E
    E --> R[Portable episode or native report]
    R --> A[Separate ACT/DP and VLA auditors]
```

| Module | Owns | Does not own |
|---|---|---|
| `data` | split, indexing, statistics, sequence alignment, cached pixels; HF/LeRobot adapters for SmolVLA and Pi-0 | simulator reset or checkpoint selection |
| `policies` | ACT, DP, Pi-0 LoRA, SmolVLA and RLT forward passes | task scheduling or simulator handles |
| `envs` | observation conversion, fixed init, action application, success predicate | learned policy weights |
| `trainers` | losses, optimizers, schedules, EMA and checkpoints | GPU selection |
| `evaluation` | histories, execution schedule, model calls and episode records | training updates |
| `runtime` | UUID/EGL mapping, checkpoint/RNG utilities, process supervision | model math |
| `workflows` | one module per recipe family: ACT/DP, HDF5 SmolVLA, official SmolVLA, HDF5 Pi-0, HF Pi-0, RLT | loss formulas |
| `reporting` | ACT/DP paired audit (`figures`) and native VLA audit (`vla`, `native`) | CUDA or simulator execution |

## ACT/DP tensor contract

- Raw images: two RGB cameras, `uint8`, CHW; ACT batch `[B,3,128,128]`, DP `[B,2,3,128,128]`.
- State: `[B,9]` or `[B,2,9]`: 7 joint positions and 2 gripper qpos.
- Task condition: integer `task_id`, `[B]`, learned embedding. No object ground-truth pose in policy inputs.
- Action: unnormalized controller command `[B,K,7]`. First six values are OSC delta commands, last value is gripper command. Clip to controller bounds at execution.
- Episode reset clears history, action queue and TE state. DP gets a dedicated random generator per task/init episode.
- `eef_pos` is recorded for analysis only; it is not passed to either policy.

## Two published evidence tracks

ACT/DP records live in `results/libero_spatial/` and use the 128×128, 9D joint, fixed-initial-state contract above. Native VLA records live in `results/vla_spatial/snapshot.json`. That snapshot checks episode coverage and success counts before drawing. It does not convert a native report into an ACT/DP initial-state id. Training curves for SmolVLA and Pi-0 share a figure only as logged losses; the axis label states that the objectives are different and are not a ranking.

`python -m roboscope report --study baseline` redraws the ACT/DP gallery. `--study vla` redraws the native gallery. `--study mujoco332` redraws the follow-up studies in `results/mujoco332/`; `--study all` redraws all three. Historical results are immutable; a simulator rerun is a new study.

## Adding a policy

Implement prediction in `policies`, a compatible observation adapter, a trainer in `trainers`, and explicit checkpoint loading in `evaluation`. Register the new route in the CLI/workflow only after end-to-end tests exist. Preserve action units in a versioned adapter rather than assuming every model emits OSC deltas. Model-specific checkpoint formats and normalization statistics must travel together.

Policy families use explicit workflows. Shared RNG, atomic checkpoint writes and batch transfer live in `runtime/training.py`; policy implementations never import a trainer. Reporting imports no model or simulator dependencies. Native SmolVLA evaluation lives in `workflows/smolvla_evaluation.py`, where checkpoint identity, task ordering and MuJoCo version are recorded before the worker starts.

## π-series and RL implications

Pi-0 adapters include language/tokenization, camera and proprioception mapping, action-unit conversion and chunk metadata. HF Spatial evaluation resolves benchmark assets by language while keeping the checkpoint’s training manifest unchanged. It uses 256px OpenGL images rotated 180 degrees; the raw HDF5 route keeps its original 128px OpenCV convention.

RL requires additional trajectory fields: reward, terminated/truncated, policy version, action/log-probability semantics, and possibly value estimates. ACT/DP chunk generation cannot be dropped into PPO as if it were a one-step Gaussian actor. Decide between a policy-native objective, residual actor, or another validated post-training method, and record behavior-policy identity before sharing this rollout layer.

Simulator vectorization is not multi-GPU DDP or asynchronous control. These capabilities have separate interfaces and validation requirements.
