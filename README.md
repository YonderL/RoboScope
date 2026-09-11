# RoboScope

**Reproducible visuomotor learning, from demonstrations to closed-loop evaluation.**

[中文](README.zh-CN.md) · [Quick start](docs/quickstart.md) · [Architecture](docs/architecture.md) · [Experiments](docs/experiments.md) · [Research findings](docs/findings.md) · [Reproduction](docs/reproducibility.md)

RoboScope implements suite-conditioned **ACT and Diffusion Policy on LIBERO-Spatial**, with fixed-initial-state evaluation, action-execution ablations, per-episode records, and reproducible figures. It is a research project: results and limitations are reported together.

![ACT and DP comparison](docs/assets/act_vs_dp.png)

## Results that changed our interpretation

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

**Not implemented:** π-series fine-tuning, PPO or other RL post-training, asynchronous/RTC inference, ManiSkill/RoboCasa, and language-conditioned policies. These are [planned extensions](docs/roadmap.md), not current benchmark claims. Checkpoints and demonstration data are not bundled.

## Reproduce the figures first

```bash
python -m pip install -e '.[report,test]'
python -m roboscope report
python -m pytest tests/test_results.py tests/test_cli.py
```

This path needs no GPU, simulator, datasets, or checkpoint download. Figures are rebuilt from [audited CSV records](results/libero_spatial/episodes.csv), not hard-coded success rates. PNG, PDF and SVG outputs are generated under `docs/assets/`.

For training and rollout evaluation, follow the [tested environment and data setup](docs/quickstart.md). Training/evaluation commands preview the plan unless `--start` is supplied. Formal runs were measured on Python 3.12, PyTorch 2.7.1+cu118 and two RTX 4090s.

## Code map

```text
src/roboscope/
  data/          # LIBERO adapter, temporal windows, normalization, image cache
  policies/      # ACT and Diffusion Policy; model and inference math
  envs/          # LIBERO setup, observation adapter, spawned environment pool
  trainers/      # ACT BC/CVAE and diffusion denoising training
  evaluation/    # matched ACT–DP protocol and historical ACT execution ablations
  runtime/       # GPU/EGL binding, worker supervision, RNG/checkpoint utilities
  workflows/     # run preparation, source snapshots, training/evaluation orchestration
  reporting/     # record validation, paired comparison, reproducible plots
configs/         # path-independent scientific recipes
examples/        # complete command sequences
results/         # small portable records and provenance, no weights or datasets
scripts/         # export/audit/release and bounded local regression tools
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
