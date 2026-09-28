# MuJoCo 3.3.2 follow-up evaluations

These evaluations use RTX PRO 5000 GPUs, PyTorch 2.7.1+cu128 and MuJoCo 3.3.2.
The official comparison merges the corrected on-ramekin task into ACT/DP and
native SmolVLA. Original episode archives remain available for provenance.
All policies have one training seed (0). Models, observation adapters and rollout
budgets differ; the numbers are not a controlled architecture comparison.

## Black bowl on the ramekin

The exact task is `pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`.
It is task **7** in the sorted HDF5/HF training manifests and task **5** in the
native LIBERO benchmark order. “Next to the ramekin” is a different task.

| Policy | Checkpoint | Execution | Historical run | MuJoCo 3.3.2 follow-up |
|---|---|---|---:|---:|
| ACT | seed 0, K8, epoch 32 | 8 actions | 15/50 (30%) | **34/50 (68%)** |
| Diffusion Policy | final EMA, 30k | DDIM10, 8 actions | 2/50 (4%) | **37/50 (74%)** |
| Native SmolVLA | 100k | 10 of 50 actions | 10/50 (20%) | **47/50 (94%)** |

Historical task counts come from `results/libero_spatial/episodes.csv` and
`results/vla_spatial/snapshot.json`. Both old and new denominators are 50;
the new outcomes replace that task in the official selection, while the old
full-suite archives are preserved.

ACT/DP preserve 128px cameras, 9D joint state, a 600-step rollout limit and
5 settling steps. Their fixed initial states are 0–49, with seed
`10000 + 1000 * task_id + initial_state_id`. SmolVLA preserves its native
256px cameras, 8D end-effector state, 280-step limit, 10 settling steps and
native seed-0 evaluation. Its report contains episode positions, not explicit
initial-state IDs; we do not label those records as paired with ACT/DP.

This PRO 5000 SmolVLA run uses the same 100k weights, BF16 and compilation
disabled. The official SmolVLA result is 411 - 10 + 47 = **448/500 (89.6%)**.
GPU, PyTorch and execution details differ across the
historical and follow-up sessions, so changes cannot be attributed solely to
MuJoCo. Concurrent GPU jobs make these runs unsuitable for latency comparisons.

## Pi-0 HF Spatial

Evaluate the completed 30k LoRA checkpoint on all ten tasks, 50 fixed initial
states per task. The adapter resolves task assets from the saved language
instructions. It uses the training data's 256px images, OpenGL followed by a
180-degree rotation, and 8D end-effector state. The checkpoint's saved
220-step limit, 10 settling steps, 50-step prediction and 8-step execution are
preserved. Training configuration and normalization are checked against the
checkpoint before inference. The completed evaluation scores **438/500 (87.6%)**.

## Official full-suite results (2026-09-28)

Each task contributes exactly 50 trials. Only on-ramekin is replaced for
ACT, DP and SmolVLA; Pi-0 LoRA uses its complete ten-task MuJoCo 3.3.2 run.
Native SmolVLA task IDs are mapped to the sorted manifest order before reporting.

| Policy | Accounting | Official SR |
|---|---|---:|
| ACT K8, epoch 32 | 415 - 15 + 34 = 434 / 500 | **86.8%** |
| DP final EMA, 30k | 409 - 2 + 37 = 444 / 500 | **88.8%** |
| SmolVLA, 100k | 411 - 10 + 47 = 448 / 500 | **89.6%** |
| Pi-0 LoRA, 30k | Full MuJoCo 3.3.2 run: 438 / 500 | **87.6%** |

[Per-task table](../results/official_spatial/per_task.csv) and
[source fingerprints](../results/official_spatial/summary.json) are generated
from the portable episode records, without copying weights or raw logs.
The ACT historical and rerun checkpoint files have different SHA-256 values;
both identities are retained with the recorded 56,064-step checkpoint label.

## Reproduce from local checkpoints

Activate the Blackwell-compatible environment with MuJoCo 3.3.2 and install
the project. Dataset/model downloads and full training are separate prerequisites.
Use fresh output directories. All entrypoints preview unless `--start` is given.

```bash
export PYTHONPATH="$PWD/src"
python -m roboscope evaluate \
  --source outputs/act_spatial_32_parallel/k008_seed0 \
  --output outputs/act_k008_mujoco332_on_ramekin_t7_pro5000 \
  --episodes 50 --task-ids 7 --checkpoint final --ta 8 --gpu-model pro5000 --start

python -m roboscope evaluate \
  --source outputs/dp_spatial_seed0 \
  --output outputs/dp_final_mujoco332_on_ramekin_t7_pro5000 \
  --episodes 50 --task-ids 7 --checkpoint final --ddim-steps 10 --ta 8 --gpu-model pro5000 --start

python -m roboscope.workflows.smolvla_evaluation \
  --source outputs/smolvla_official_spatial_seed0 --checkpoint 100000 \
  --libero-root LIBERO/libero/libero \
  --output outputs/smolvla_mujoco332_on_ramekin_t5_pro5000 \
  --task-ids 5 --episodes 50 --action-steps 10 --gpu-model pro5000 --mujoco-version 3.3.2 --start

python -m roboscope evaluate \
  --source outputs/pi0_lora_hf_spatial_pro5000_seed0 \
  --output outputs/pi0_hf_mujoco332_eval50_pro5000 \
  --libero-root LIBERO/libero/libero --episodes 50 --gpu-model pro5000 --start

# Export only after every study has completed; malformed/incomplete shards fail.
python scripts/export_evaluation_results.py
python scripts/export_official_results.py
python -m roboscope report --study all
```

The portable publication includes trial outcomes, checkpoint fingerprints,
scientific configuration and source checksums. Local paths are replaced by
placeholders; provide your own assets. Logs, weights, videos, trajectory arrays,
smoke tests and superseded diagnostics are not uploaded.
