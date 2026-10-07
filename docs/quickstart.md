# Quick start

## 1. Figures only (CPU)

From the repository root:

```bash
python -m pip install -e '.[report,test]'
python -m roboscope report --study all
python -m pytest tests/test_results.py tests/test_cli.py
```

No training stack is imported by the reporting CLI. `results/` contains the portable records for every published chart. Base package installation deliberately does not upgrade PyTorch.

## 2. Training environment

The measured runs used Linux, Python 3.12, PyTorch **2.7.1+cu118**, torchvision **0.22.1+cu118**, LeRobot **0.6.1**, diffusers **0.39.0**, hf-libero **0.1.4**, robosuite **1.4.0**, MuJoCo **3.8.1**. Existing users should keep the working `lerobot` environment. The report does not require a driver upgrade.

For a new environment, the intended installation sequence is:

```bash
conda create -n roboscope python=3.12 -y
conda activate roboscope
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -c constraints-lerobot.txt 'cmake>=3.29,<4' setuptools wheel
# Old EGL probe packages need access to the installed CMake during their build.
python -m pip install --no-build-isolation egl_probe==1.0.2 hf-egl-probe==1.0.2
python -m pip install -c constraints-lerobot.txt -r requirements-training.txt
python -m pip install -e '.[test]'
python -m pip check
```

This is a pinned installation recipe, not a claim that a clean machine was provisioned in this refactor. EGL probe compilation additionally needs a working C/C++ compiler and build tools. Avoid `lerobot[all]`: it installs unrelated hardware integrations. Both OpenCV wheels may be pulled by upstream requirements; the code consumes RGB arrays and does not use OpenCV for preprocessing.

Check the runtime before allocating a long job:

```bash
python -c "import torch; from roboscope.policies.act import TaskACT; from roboscope.policies.diffusion import TaskDiffusionPolicy; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

## 3. Data and benchmark assets

Download the official LIBERO-Spatial HDF5 demonstrations and LIBERO benchmark assets following the [benchmark repository](https://github.com/Lifelong-Robot-Learning/LIBERO). You need:

```text
DATA_ROOT/libero_spatial/*_demo.hdf5       # 10 task files
LIBERO_ROOT/bddl_files/libero_spatial/
LIBERO_ROOT/init_files/libero_spatial/
LIBERO_ROOT/assets/
```

`LIBERO_ROOT` usually ends in `LIBERO/libero/libero`. Use the existing checkout for assets; the simulator Python distribution used here is `hf-libero`. Do not install a second conflicting `libero` module in the same environment. Do not execute from inside the LIBERO checkout.

These recipes expect original 128×128 dual-camera data, 9D proprioception and 7D OSC_POSE actions. They are not automatically compatible with re-rendered/no-op-filtered OpenVLA datasets. Split seed 2026 produces 45 train / 5 validation demonstrations per task for the measured data. Task/init file hashes and split membership are published in `results/libero_spatial/protocol.json`.

## 4. Train a baseline

Replace the example paths with your local dataset and assets. No absolute user paths are embedded in recipes.

```bash
# Preview only; no data read or GPU initialization.
roboscope train --recipe configs/libero_spatial/act_baseline.json \
  --data-root /path/to/data --libero-root /path/to/LIBERO/libero/libero \
  --output outputs/act_baseline

# Add --start to execute. ACT also evaluates at epochs 8/16/24/32.
roboscope train --recipe configs/libero_spatial/act_baseline.json \
  --data-root /path/to/data --libero-root /path/to/LIBERO/libero/libero \
  --output outputs/act_baseline --start

# DP is independent of an existing ACT run; same deterministic split.
roboscope train --recipe configs/libero_spatial/diffusion.json \
  --data-root /path/to/data --libero-root /path/to/LIBERO/libero/libero \
  --output outputs/dp --start
```

Add `--resume` after interruption; configuration, dataset manifest and source must match. DP trains one multi-task model on one 4090. ACT ablation uses `act_ablation.json` and schedules independent runs on both cards. Current GPU routing intentionally requires exactly two RTX 4090s and refuses fallback to other models. General device selection is a future runtime extension.

## 5. Evaluate ACT and DP with the same 50 initial states

```bash
roboscope evaluate --source outputs/act_baseline/k008_seed0 \
  --output outputs/eval_act --checkpoint final --episodes 50 --ta 8 --start
roboscope evaluate --source outputs/dp \
  --output outputs/eval_dp_final --checkpoint final --episodes 50 --ddim-steps 10 --ta 8 --start
roboscope evaluate --source outputs/dp \
  --output outputs/eval_dp_best --checkpoint best --episodes 50 --ddim-steps 10 --ta 8 --start
```

Each command uses two GPUs, 5 tasks per shard, 8 environments per GPU. Run commands sequentially, not simultaneously. Each produces `eval/<variant>/shard*/{config.json,episodes.jsonl,complete.json,latency.json}`, videos/traces, and `summary.json`. Give each checkpoint / inference variant a distinct output. A resume retains completed episodes and verifies checkpoint identities. A changed inference batch schedule can still change floating-point actions; see [reproducibility](reproducibility.md).

The `report` command redraws the bundled study; it does **not** silently substitute new runs into published figures. New studies should export and audit their records under a new result directory.
