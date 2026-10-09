# RoboScope

**A reproducible research workbench for learning and evaluating robot policies on LIBERO-Spatial.**

[简体中文](README.zh-CN.md) · [Quick start](docs/quickstart.md) · [Experiment protocols](docs/experiments.md) · [Results and episode records](results/README.md) · [Research findings](docs/findings.md)

RoboScope connects demonstrations, policy training, closed-loop simulation, and auditable results. I built it to answer a practical research question: **when a robot policy's success rate changes, was it the model, the checkpoint, the action schedule, or the evaluation protocol?** The repository implements ACT, Diffusion Policy, SmolVLA, Pi-0 LoRA, and an RL Token post-training route. A separate [RLinf SmolVLA+PPO branch in my fork](https://github.com/YonderL/RLinf/tree/codex/smolvla-ppo) contains the PPO adapter and its LIBERO examples.

**4 policy families · 2 RL post-training routes · 10 tasks × 50 trials · 3,500 audited ACT/DP episodes**

![Per-task LIBERO-Spatial closed-loop results for ACT, Diffusion Policy, SmolVLA and Pi-0 LoRA](docs/assets/policy_success_rates.png)

## What I built

| Area | Contribution | Where to inspect it |
|---|---|---|
| **Training** | Multi-task ACT and Conditional 1D U-Net Diffusion Policy pipelines; official-data SmolVLA SFT and Pi-0 LoRA workflows; checkpointing and resume. The ACT study spans 4 chunk sizes × 3 seeds. | [Recipes](configs/libero_spatial/) · [Workflows](src/roboscope/workflows/) |
| **Data and runtime** | Trajectory-level train/validation splits, train-only normalization, image-convention checks, temporal windows, cached image loading, GPU/EGL binding, and isolated experiment snapshots. | [Data modules](src/roboscope/data/) · [Runtime](src/roboscope/runtime/) |
| **Closed-loop evaluation** | Fixed initial states, per-episode action/history state, chunk execution, per-step replanning and ACT temporal ensemble; explicit task mapping, checkpoint identity and simulator protocol. | [Evaluation](src/roboscope/evaluation/) · [Protocol](docs/reproducibility.md) |
| **Evidence and analysis** | Audited 3,500 portable ACT/DP episode records, checkpoint/inference ablations, paired analyses, and figures reproducible on a CPU without downloading weights or a simulator. | [Episode records](results/libero_spatial/episodes.csv) · [Findings](docs/findings.md) |
| **RL post-training** | A frozen-VLA RL Token route with sparse and progress rewards; a SmolVLA PPO adapter integrated into RLinf and evaluated separately. | [RLT method](docs/smolvla_rlt_hf.md) · [PPO code in my RLinf fork](https://github.com/YonderL/RLinf/tree/codex/smolvla-ppo) |

The model architectures and benchmark are existing research; this project's contribution is the training/evaluation integration, experimental controls, and released evidence. [Architecture and module boundaries](docs/architecture.md).

## Closed-loop results

The four baseline families below were each evaluated on all 10 LIBERO-Spatial tasks, with 50 episodes per task. These are the repository's **official selected results**, including corrected MuJoCo 3.3.2 reruns for the “on the ramekin” task. Each family retains its own observation, action and rollout settings, so this table shows completed workflows **rather than an architecture-only ranking**.

| Policy and selected checkpoint | Successes | SR | Recipe |
|---|---:|---:|---|
| ACT · K=8, epoch 32 | 434/500 | **86.8%** | [ACT](configs/libero_spatial/act_baseline.json) |
| Diffusion Policy · DDIM=10, Ta=8, final 30k | 444/500 | **88.8%** | [DP](configs/libero_spatial/diffusion.json) |
| SmolVLA · official Spatial SFT 100k, execute 10/50 | 448/500 | **89.6%** | [SmolVLA](configs/libero_spatial/smolvla_official.json) |
| Pi-0 LoRA · HF Spatial 30k, execute 8 | 438/500 | **87.6%** | [Pi-0](configs/libero_spatial/pi0_lora_hf_spatial.json) |

[Per-task counts, source selection and checkpoint hashes](results/official_spatial/summary.json) · [Rerun protocol](results/mujoco332/protocol.json) · [Evaluation accounting](docs/evaluation_20260926.md). The first three rows combine their archived nine tasks with the corrected on-ramekin rerun; Pi-0 uses a complete MuJoCo 3.3.2 evaluation. All are single-training-seed results.

### A finding that changed checkpoint selection

In the **separate matched ACT/DP study**, Diffusion Policy's lowest-validation-noise-MSE checkpoint at 7k updates achieved **347/500 (69.4%)**, while the final 30k checkpoint achieved **409/500 (81.8%)** under the same 500 task/initial-state pairs. Selecting the final checkpoint rescued 93 episodes and lost 31, a net **+12.4 percentage points without retraining**. This shows why an offline denoising loss alone did not select the stronger closed-loop controller in this run. The follow-up checkpoint comparison was exploratory; the paired bootstrap and its limits are in [the full analysis](docs/findings.md).

![Diffusion Policy checkpoint selection: validation loss and closed-loop success](docs/assets/checkpoint_selection.png)

## SmolVLA post-training: two routes

Both routes start from the official Spatial SFT 100k checkpoint, but have different algorithms and runtimes. The SFT reference here is the **C=10 post-training control (442/500)**, not the 448/500 official-table selection above.

| Route | Implementation | LIBERO-Spatial result | Interpretation |
|---|---|---:|---|
| SFT control, C=10 | [RLT comparison record](results/rlt_hf_spatial/summary.json) | 442/500 · 88.4% | Reference for post-training comparisons |
| RL Token + progress reward | [RoboScope RLT workflow](docs/smolvla_rlt_spatial.md) | **462/500 · 92.4%** | +4.0 pp versus its matched SFT control |
| PPO, iteration 100 | [My RLinf fork: `codex/smolvla-ppo`](https://github.com/YonderL/RLinf/tree/codex/smolvla-ppo) | **449/500 · 89.8%** | +1.4 pp versus historical SFT; one training seed |

RLT freezes SmolVLA and trains a Gaussian actor with twin critics; its progress reward uses reach, grasp and transport signals only for reward computation. The PPO adapter instead updates SmolVLA's action expert using RLinf's actor-critic loop. Its completed run used 100 rollout/update iterations and a BF16, 500-episode final evaluation. On the top-drawer task, PPO scored 43/50; a same-BF16-runtime SFT check scored 42/50. The 50-episode evaluations during training ended at 40/50 for both PPO and their SFT baseline; those checks use a different protocol from the full evaluation. [RLT results](results/rlt_hf_spatial/summary.json) · [PPO protocol, parameters and 500 episode records](docs/smolvla_ppo_spatial.md) · [PPO Spatial configuration in my RLinf fork](https://github.com/YonderL/RLinf/blob/codex/smolvla-ppo/examples/embodiment/config/libero_spatial_ppo_smolvla.yaml).

The +1.4 pp PPO observation is small and descriptive. Historical SFT and PPO use different LeRobot/Transformers model runtimes; a full same-runtime SFT evaluation and additional training seeds are needed to assess stability. PPO training code lives in the linked RLinf fork, **not** in RoboScope's trainer package.

## Reproduce the evidence first

Figures and published-record checks run on a CPU; no GPU, LIBERO installation, dataset or checkpoint is required:

```bash
python -m pip install -e '.[report,test]'
python scripts/export_official_results.py
python -m roboscope report --study all
python -m pytest tests/test_results.py tests/test_native_results.py \
  tests/test_followup_results.py tests/test_official_results.py tests/test_cli.py
```

The ACT/DP comparison is backed by [episode-level CSV records](results/libero_spatial/episodes.csv); post-training has separate [RLT](results/rlt_hf_spatial/summary.json) and [PPO](results/smolvla_ppo_spatial/summary.json) records. To train or run simulator evaluations, follow the [environment and data setup](docs/quickstart.md). Training commands preview their plan until `--start` is supplied. Checkpoints, demonstrations, simulator assets, raw videos and traces are not bundled. [Reproducibility contract](docs/reproducibility.md) · [Validation record](docs/validation.md).

## Repository map

```text
src/roboscope/   data, policies, trainers, evaluation, runtime, reporting, workflows
configs/         versioned scientific recipes and the RLinf Spatial override
results/         compact episode evidence, task summaries and provenance
scripts/         training/export entrypoints and source-release builder
docs/            protocols, analysis, runbooks and regenerated figures
```

RoboScope keeps policy-specific training paths explicit and shared infrastructure limited to data, runtime and reporting contracts. It does not claim distributed PPO support in this repository. [Code architecture](docs/architecture.md) · [Contributing](CONTRIBUTING.md) · [Apache-2.0 license](LICENSE) · [Third-party notices](THIRD_PARTY_NOTICES.md).

Built with [LeRobot](https://github.com/huggingface/lerobot) and [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO); please cite the original ACT, Diffusion Policy, SmolVLA, Pi-0 and benchmark work when using their methods or data.
