# RoboScope

**Reproducible visuomotor learning, from demonstrations to closed-loop evaluation.**

[中文](README.zh-CN.md) · [Quick start](docs/quickstart.md) · [Architecture](docs/architecture.md) · [Experiments](docs/experiments.md) · [Research findings](docs/findings.md) · [Reproduction](docs/reproducibility.md)

RoboScope implements suite-conditioned **ACT and Diffusion Policy on LIBERO-Spatial**, with fixed-initial-state evaluation, action-execution ablations, per-episode records, and reproducible figures. It also provides a **language-conditioned Pi-0 LoRA fine-tuning pipeline for two RTX 4090s**; see the [Pi-0 runbook and validation limits](docs/pi0_lora_spatial.md). It is a research project: results and limitations are reported together.

![ACT and DP comparison](docs/assets/act_vs_dp.png)

## Current closed-loop scores

Each cell is successes out of 50 fixed trials. Rows are the same LIBERO-Spatial task, matched by name. ACT and DP use their HDF5 task order; SmolVLA uses the native benchmark order, where “on the ramekin” is task 5 rather than task 7. Pi-0 uses the same names as ACT and DP.

| Task | ACT | DP | SmolVLA | Pi-0 |
|---|---:|---:|---:|---:|
| Between plate and ramekin | 42/50 | 48/50 | 47/50 | 50/50 |
| Table center | 48/50 | 50/50 | 49/50 | 49/50 |
| Top drawer | 45/50 | 45/50 | 44/50 | 46/50 |
| Next to cookie box | 48/50 | 47/50 | 49/50 | 47/50 |
| Next to plate | 38/50 | 35/50 | 38/50 | 37/50 |
| Next to ramekin | 41/50 | 49/50 | 47/50 | 42/50 |
| On cookie box | 46/50 | 47/50 | 45/50 | 45/50 |
| On the ramekin | 34/50 | 37/50 | 47/50 | 42/50 |
| On the stove | 43/50 | 44/50 | 42/50 | 38/50 |
| On the wooden cabinet | 49/50 | 42/50 | 40/50 | 42/50 |
| **Suite** | **434/500 (86.8%)** | **444/500 (88.8%)** | **448/500 (89.6%)** | **438/500 (87.6%)** |

ACT is K=8 chunk execution at epoch 32. DP is the 30k final checkpoint, DDIM=10, Ta=8. SmolVLA is the 100k official Spatial checkpoint, executing 10 of 50 predicted actions. Pi-0 is the 30k HF Spatial LoRA checkpoint, executing 8 actions, on MuJoCo 3.3.2 for every task.

The “on the ramekin” row is the MuJoCo 3.3.2 re-evaluation on PRO 5000s for ACT, DP, and SmolVLA. Their other nine tasks stay on the earlier published runs. Pi-0’s whole table is the new MuJoCo 3.3.2, 256px, 8D-state evaluation. Image size, proprioception, and rollout budget still differ, so this is a side-by-side of the current scores, not one shared protocol.

## Results that changed our interpretation

The original matched ACT/DP study, before the ramekin re-evaluation above:

| Policy | Checkpoint | Success / 500 | SR | Inference p50 |
|---|---|---:|---:|---:|
| ACT, K=8, chunk execution | Epoch 32 | 415 | **83.0%** | 12.0 ms |
| DP, DDIM=10, Ta=8 | Final, 30k updates | 409 | **81.8%** | 66.2 ms |
| DP, DDIM=10, Ta=8 | Minimum validation noise MSE, 7k | 347 | 69.4% | 63.0 ms |

Same seed 0, ten tasks, 50 fixed initial states per task, two RGB cameras, 7D OSC_POSE actions, and rollout budget. Visual encoders, observation history, normalization and training budgets differ: **this is a policy-recipe comparison, not an architecture-only ablation**. Latency is batch-1 full prediction, including preprocessing and action D2H, excluding input H2D, simulation and network.

**Checkpoint selection changed DP success by +12.4 percentage points without retraining.** Validation denoising loss increased while closed-loop performance improved. DP final is only 1.2 points below ACT; one training seed does not establish a stable ranking. Final was evaluated after inspecting the earlier result, so checkpoint analysis is exploratory. [Full analysis and paired uncertainty](docs/findings.md).

## What is implemented

- **Policies:** task-ID-conditioned ACT and CNN / Conditional 1D U-Net Diffusion Policy, using LeRobot model components.
- **Data:** trajectory-level train/validation split, explicit image convention, train-only normalization, frame and temporal-window datasets, memory-mapped image cache and CUDA prefetch.
- **Execution:** chunk execution, per-step replanning, ACT temporal ensemble; independent state per episode. Parallel simulators with synchronous policy inference.
- **Experiments:** 12 ACT models (four chunk sizes × three seeds), checkpoint curves, DP sampler/execution-horizon ablations, best-versus-final analysis.
- **Evidence:** 3,500 portable episode records for the ACT–DP study, checkpoint hashes, task/init identities, training history, and CPU-only figure regeneration. Historical ACT curves retain their separate three-seed / 20-episode protocol.
- **Runtime:** two RTX 4090s selected by UUID, matched to EGL via PCI address; isolated output directories and source snapshots.

**SmolVLA, official Spatial protocol:** paper architecture on the released 256×256 dataset, 8D end-effector state, 100k updates, seed 0. Executing 10 of 50 predicted actions scores **448/500 (89.6%)** after replacing the “on the ramekin” task with the MuJoCo 3.3.2 re-evaluation (47/50; the earlier run of that task was 10/50). In-training rollouts replan every step and are a different number (68–73% on 100 episodes). Recipe: [smolvla_official.json](configs/libero_spatial/smolvla_official.json). Per-task scores are in the table above.

![SmolVLA native evaluation](docs/assets/smolvla_evaluation.png)

**Pi-0 LoRA, HF Spatial:** 30k updates, seed 0, evaluated for 50 trials on every task under MuJoCo 3.3.2. Closed-loop success is **438/500 (87.6%)**. The training curve is below. Recipe: [pi0_lora_hf_spatial.json](configs/libero_spatial/pi0_lora_hf_spatial.json). The older HDF5 [Pi-0 runbook](docs/pi0_lora_spatial.md) is a separate path and is not the score in the table.

![VLA training evidence](docs/assets/vla_training.png)

**Other SmolVLA paths, without a published success rate:** the HDF5 recipe still targets the shared ACT/DP evaluator ([smolvla_spatial.md](docs/smolvla_spatial.md)). RLT post-training adds a learned RL token and a Gaussian actor/twin critic on frozen SFT features, executes 10 steps, and keeps a matched 10-step SFT control ([smolvla_rlt_spatial.md](docs/smolvla_rlt_spatial.md)). Bounded simulator tests pass; success-rate gains have not been measured. Preview with `bash scripts/posttrain_smolvla_rlt_spatial.sh --preview`.

Run the stages separately with `--stage token`, `--stage warmup`, `--stage online`, and `--stage evaluate`. Freeze the RL token before collecting feature replay; online learning continues collecting new rollouts. See the [step-by-step commands](docs/smolvla_rlt_spatial.md#按阶段执行).

**Not implemented:** PPO, asynchronous/RTC inference, and ManiSkill/RoboCasa. These are [planned extensions](docs/roadmap.md), not current benchmark claims. Checkpoints and demonstration data are not bundled.

## Reproduce the figures first

```bash
python -m pip install -e '.[report,test]'
python -m roboscope report --study all
python -m pytest tests/test_results.py tests/test_native_results.py tests/test_cli.py
```

This path needs no GPU, simulator, datasets, or checkpoint download. ACT/DP figures are rebuilt from [audited CSV records](results/libero_spatial/episodes.csv). VLA figures are rebuilt from [results/vla_spatial/snapshot.json](results/vla_spatial/snapshot.json). Success rates are checked against the episode records. PNG, PDF and SVG outputs are generated under `docs/assets/`.

For training and rollout evaluation, follow the [tested environment and data setup](docs/quickstart.md). Training/evaluation commands preview the plan unless `--start` is supplied. Formal runs were measured on Python 3.12, PyTorch 2.7.1+cu118 and two RTX 4090s.

## Code map

```text
src/roboscope/
  data/          # LIBERO HDF5 adapter, HF/LeRobot adapters, normalization, caches
  policies/      # ACT, Diffusion Policy, Pi-0 LoRA, SmolVLA, RLT actor
  envs/          # LIBERO setup, observation adapter, spawned environment pool
  trainers/      # behavior cloning, diffusion, official SmolVLA, Pi-0, RLT
  evaluation/    # matched ACT–DP protocol and HF Pi-0 closed-loop adapter
  runtime/       # GPU/EGL binding, checkpoint/RNG utilities, worker supervision
  workflows/     # one entry per recipe family; no shared loss formulas
  reporting/     # ACT/DP paired audit and a separate native VLA audit
configs/         # path-independent scientific recipes
examples/        # complete command sequences
results/         # baseline, VLA training, and MuJoCo 3.3.2 follow-up records
scripts/         # export, train entrypoints, release allowlist
```

Structure is inspired by [verl-vla](https://github.com/verl-project/verl-vla)'s separation of workflows, training, and integrations. RoboScope does **not** depend on verl or claim its distributed/RL capabilities.

## Research gallery

![Checkpoint selection](docs/assets/checkpoint_selection.png)

![ACT learning curves](docs/assets/act_learning_curves.png)

[DP inference ablations on the 7k checkpoint](docs/assets/dp_best_ablations.png) · [Protocol and configuration matrix](docs/experiments.md)

## Reproduction status

Published metrics come from the archived pre-refactor experiment implementation. The packaged implementation preserves model math and evaluation scheduling; the validation record describes what has actually been tested after refactoring. Re-running an entire benchmark after refactoring is a separate validation step, not implied by import/unit tests. See [validation](docs/validation.md) and [migration](docs/migration.md).

## Contributing and attribution

[Contributing](CONTRIBUTING.md) · [GitHub release guide](docs/github_release.md) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Apache-2.0 license](LICENSE)

Built on [LeRobot](https://github.com/huggingface/lerobot), [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [ACT](https://github.com/tonyzhaozh/act), and [Diffusion Policy](https://github.com/real-stanford/diffusion_policy). Please cite the original methods and benchmark when using this project. Project software version: **0.1.0**.
