"""Publish only completed key evaluations, portable configs and per-trial outcomes."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from roboscope.evaluation.pi0 import aggregate_evaluation
from roboscope.reporting.native import native_episodes
from roboscope.reporting.records import read_evaluation

TASK = "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def portable_config(value):
    """Keep scientific settings; local filesystem locations are supplied at runtime."""
    if isinstance(value, dict):
        return {k: portable_config(v) for k, v in value.items()}
    if isinstance(value, list):
        return [portable_config(v) for v in value]
    if isinstance(value, str) and value.startswith("/"):
        return "<local-path>"
    return value


def export(workspace, output):
    base = Path(workspace) / "outputs"
    output = Path(output)
    records, studies = [], {}

    def append(policy, rows, checkpoint, native=False):
        for row in rows:
            records.append(
                {
                    "policy": policy,
                    "task_id": row["task_id"],
                    "task_name": row.get("task_name", TASK),
                    "trial_index": row["episode_index" if native else "initial_state_id"],
                    "identity_kind": "native_episode_index" if native else "fixed_initial_state",
                    "eval_seed": "" if native else row["eval_seed"],
                    "success": row["success"],
                    "steps": row.get("steps", ""),
                    "checkpoint_sha256": checkpoint,
                }
            )

    for policy, run, variant in [
        ("act", "act_k008_mujoco332_on_ramekin_t7_pro5000", "act_k008_chunk"),
        ("dp", "dp_final_mujoco332_on_ramekin_t7_pro5000", "dp_ddim10_ta08"),
    ]:
        root = base / run
        rows, identity = read_evaluation(root / "eval" / variant, 50, [7])
        append(policy, rows, identity["checkpoint_sha256"])
        config = json.loads((root / "config.json").read_text())
        task = next(t for t in json.loads((root / "manifest.json").read_text())["tasks"] if t["id"] == 7)
        if task["name"] != TASK:
            raise ValueError("Wrong ramekin task")
        studies[policy] = {
            "task_ids": [7],
            "episodes_per_task": 50,
            "checkpoint_sha256": identity["checkpoint_sha256"],
            "gradient_step": identity["gradient_step"],
            "training_recipe": f"configs/libero_spatial/{'act_baseline' if policy == 'act' else 'diffusion'}.json",
            "evaluation_config": portable_config(config),
            "benchmark": {k: task[k] for k in ("name", "bddl_sha256", "init_sha256", "controller")},
            "source_sha256": {
                f"shard{i}/{name}": digest(root / "eval" / variant / f"shard{i}" / name)
                for i in range(2)
                for name in ("config.json", "complete.json")
            },
            "episode_source_sha256": digest(root / "eval" / variant / "shard1/episodes.jsonl"),
            "environment_note": "MuJoCo 3.3.2, PyTorch 2.7.1+cu128; PRO 5000. ACT launcher recorded simulator version; DP used the same installed environment.",
        }
    root = base / "smolvla_mujoco332_on_ramekin_t5_pro5000"
    report = root / "evaluation/eval_info.json"
    identity = json.loads((root / "identity.json").read_text())
    if identity["task_names"] != [TASK] or identity["environment"]["mujoco"] != "3.3.2":
        raise ValueError("Wrong native task or simulator")
    rows = native_episodes([json.loads(report.read_text())], [5], 50)
    append("smolvla", rows, identity["checkpoint_sha256"], native=True)
    studies["smolvla"] = {
        "task_ids": [5],
        "episodes_per_task": 50,
        "checkpoint_sha256": identity["checkpoint_sha256"],
        "gradient_step": int(identity["checkpoint_step"]),
        "training_recipe": "configs/libero_spatial/smolvla_official.json",
        "evaluation_config": portable_config(identity["evaluation_config"]),
        "policy_config": portable_config(json.loads((root / "checkpoint/config.json").read_text())),
        "environment": identity["environment"],
        "action_steps": identity["action_steps"],
        "source_sha256": {"eval_info.json": digest(report), "identity.json": digest(root / "identity.json")},
        "identity_note": "Native report order, no explicit saved initial-state IDs; not paired with ACT/DP trials.",
    }
    root = base / "pi0_hf_mujoco332_eval50_pro5000"
    summary = aggregate_evaluation(root)
    identity = json.loads((root / "evaluation_identity.json").read_text())
    if summary["smoke"] or summary["episodes"] != 500 or identity["environment"]["mujoco"] != "3.3.2":
        raise ValueError("Only the complete 500-episode Pi-0 evaluation can be published")
    rows = [
        json.loads(line)
        for shard in range(2)
        for line in (root / f"shard_{shard}/episodes.jsonl").read_text().splitlines()
    ]
    append("pi0", rows, summary["checkpoint_sha256"])
    studies["pi0"] = {
        "task_ids": list(range(10)),
        "episodes_per_task": 50,
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "gradient_step": 30000,
        "training_recipe": "configs/libero_spatial/pi0_lora_hf_spatial.json",
        "protocol": summary["protocol"],
        "environment": identity["environment"],
        "benchmark_tasks": [
            {k: t[k] for k in ("id", "name", "bddl_sha256", "init_sha256")}
            for t in identity["benchmark_tasks"]
        ],
        "source_sha256": {
            f"shard_{i}/episodes.jsonl": digest(root / f"shard_{i}/episodes.jsonl") for i in range(2)
        },
    }
    # No logs, videos, trajectories, latencies, model weights or machine paths.
    output.mkdir(parents=True, exist_ok=True)
    with (output / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "date": "2026-09-26",
                "suite": "libero_spatial",
                "studies": studies,
                "limitations": [
                    "One trained seed per policy. Trial variability is not training-seed uncertainty.",
                    "ACT/DP and SmolVLA use different observations, rollout limits and episode protocols.",
                    "Follow-up GPU and simulator versions differ from historical runs; changes are not isolated causal effects of MuJoCo.",
                    "Concurrent GPU jobs may affect timing; inference latency is not published for these runs.",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("results/mujoco332"))
    args = parser.parse_args()
    export(args.workspace, args.output)
