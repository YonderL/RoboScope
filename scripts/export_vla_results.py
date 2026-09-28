"""Export a fixed, audited snapshot of completed VLA runs; no GPU or model loading."""

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

from roboscope.reporting.native import audit_snapshot, native_episodes


def sha(path):
    with path.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
        return digest.hexdigest()


def export(workspace):
    workspace = Path(workspace)
    smol = workspace / "outputs/smolvla_official_spatial_seed0"
    pi0 = workspace / "outputs/pi0_lora_hf_spatial_pro5000_seed0"
    sources = {}

    def read(path):
        sources[str(path.relative_to(workspace))] = sha(path)
        return path.read_text()

    checkpoint = smol / "eval_exec10_100k/checkpoint/model.safetensors"
    checkpoint_sha = sha(checkpoint)
    original = smol / "training/checkpoints/100000/pretrained_model/model.safetensors"
    if sha(original) != checkpoint_sha:
        raise ValueError("Execution ablation must use the same 100k model weights")
    policy = json.loads(read(checkpoint.with_name("config.json")))
    if (policy["n_action_steps"], policy["chunk_size"]) != (10, 50):
        raise ValueError("Expected 50 predicted / 10 executed actions")
    evaluations = []
    for directory, count in [("eval_exec10_100k_50", 50)]:
        reports = []
        for shard in ("a", "b"):
            config = json.loads(read(smol / directory / f"eval_{shard}.json"))
            if config["eval"]["n_episodes"] != count or config["seed"] != 0:
                raise ValueError("Unexpected evaluation protocol")
            reports.append(json.loads(read(smol / directory / shard / "eval_info.json")))
        evaluations.append(
            {
                "name": f"smolvla_100k_exec10_{count}ep",
                "checkpoint_step": 100000,
                "checkpoint_sha256": checkpoint_sha,
                "task_ids": list(range(10)),
                "episodes_per_task": count,
                "episodes": native_episodes(reports, range(10), count),
                "protocol": {
                    "training_seed": 0,
                    "evaluation_seed": 0,
                    "chunk_size": 50,
                    "execution_horizon": 10,
                    "image_size": 256,
                    "state_dim": 8,
                    "rollout_horizon": 280,
                    "settle_steps": 10,
                    "episode_identity": "Native report order; no recorded initial-state IDs or per-episode seeds",
                    "simulator": "MuJoCo 3.8.1 per archived training environment; evaluation did not save a separate environment lock",
                },
            }
        )
    log = read(smol / "train.log")
    losses, periodic = {}, {}
    for line in log.splitlines():
        if "loss:" in line:
            progress = re.search(r"(\d+)/([0-9]+)\s+\[", line)
            loss = re.search(r"\bloss:([0-9.e+-]+)", line)
            if progress and loss:
                # tqdm retains exact counts where the logger abbreviates 99,900 to 100K.
                step = 100000 - int(progress[2]) + int(progress[1])
                losses[step] = {"step": step, "train_loss": float(loss[1])}
        if "Suite overall aggregated:" in line:
            step = re.search(r"videos_step_(\d+)", line)
            rate = re.search(r"'pc_success': ([\d.]+)", line)
            count = re.search(r"'n_episodes': (\d+)", line)
            if not (step and rate and count):
                raise ValueError("Periodic evaluation has no step/rate/count")
            periodic[int(step[1])] = {
                "step": int(step[1]),
                "success_rate": float(rate[1]),
                "episodes": int(count[1]),
            }
    if max(losses, default=0) != 100000:
        raise ValueError("Incomplete SmolVLA training log")
    metrics = list(csv.DictReader(read(pi0 / "train_metrics.csv").splitlines()))
    rows = [
        {
            "step": int(r["step"]),
            "train_loss": float(r["train_loss"]),
            "validation_loss": float(r["validation_loss"]) if r["validation_loss"] else None,
        }
        for r in metrics
        if int(r["step"]) % 100 == 0 or r["validation_loss"]
    ]
    if rows[-1]["step"] != 30000:
        raise ValueError("Incomplete Pi-0 training metrics")
    return audit_snapshot(
        {
            "schema": 1,
            "captured_on": "2026-09-26",
            "source_sha256": sources,
            "evaluations": evaluations,
            "training": {
                "smolvla": {
                    "rows": sorted(losses.values(), key=lambda r: r["step"]),
                    "note": "Logged minibatch loss, rounded to 3 decimals. Exact steps from tqdm; last resumed observation wins.",
                },
                "pi0": {
                    "rows": rows,
                    "note": "HF Spatial LoRA, seed 0, every 100 steps plus validation; rollouts are a separate study.",
                },
            },
            "periodic_evaluation": sorted(periodic.values(), key=lambda r: r["step"]),
            "limitations": [
                "Single training seed; no training-seed uncertainty is established.",
                "Native protocol differs from historical ACT/DP; no matched leaderboard or paired inference.",
                "Checkpoint hash checked at export, not recorded by native evaluator at rollout time.",
                "Periodic evaluation uses execution horizon 1; standalone ablations use 10.",
            ],
        }
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/vla_spatial/snapshot.json"))
    args = parser.parse_args()
    snapshot = export(args.workspace)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(snapshot, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
