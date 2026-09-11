"""Publish small, path-free records from completed local runs. Never run a policy."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

from roboscope.reporting.records import audit_portable, read_evaluation


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, default=Path("results/libero_spatial"))
    a = p.parse_args()
    base = a.workspace / "outputs/dp_spatial_seed0"
    final = a.workspace / "outputs/dp_spatial_final_comparison"
    specs = [
        ("act_final", base, "act_k008_chunk"),
        ("dp_best", base, "dp_ddim10_ta08"),
        ("dp_final", final, "dp_ddim10_ta08"),
    ]
    specs += [
        (f"dp_best_{v}", base, v)
        for v in ["dp_ddim05_ta08", "dp_ddim20_ta08", "dp_ddim10_ta01", "dp_ddim10_ta04"]
    ]
    output = a.output
    output.mkdir(parents=True, exist_ok=True)
    exports = []
    metadata = {}
    import statistics

    for label, root, variant in specs:
        directory = root / "eval" / variant
        rows, identity = read_evaluation(directory)
        samples = []
        proof = []
        for i in range(2):
            shard = directory / f"shard{i}"
            latency = json.loads((shard / "latency.json").read_text())
            samples += latency["samples_ms"]
            proof.append(
                {
                    "shard": i,
                    "episodes_sha256": sha(shard / "episodes.jsonl"),
                    "config_sha256": sha(shard / "config.json"),
                }
            )
        metadata[label] = {
            **identity,
            "latency_samples_ms": samples,
            "latency_p50_ms": statistics.median(samples),
            "source_files": proof,
        }
        for row in rows:
            exports.append(
                {
                    "experiment": label,
                    **{
                        k: row[k]
                        for k in (
                            "task_id",
                            "task_name",
                            "initial_state_id",
                            "eval_seed",
                            "success",
                            "steps",
                            "model_calls",
                            "checkpoint_sha256",
                        )
                    },
                }
            )
    audit_portable(exports)
    with (output / "episodes.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(exports[0]))
        writer.writeheader()
        writer.writerows(exports)
    manifest = json.loads((base / "manifest.json").read_text())
    # Only portable task identities and split membership are distributed, no user paths.
    tasks = [
        {
            k: t[k]
            for k in (
                "id",
                "name",
                "episodes",
                "bddl_sha256",
                "init_sha256",
                "image_convention",
                "controller",
            )
        }
        for t in manifest["tasks"]
    ]
    (output / "protocol.json").write_text(
        json.dumps(
            dict(
                suite="libero_spatial",
                training_seeds=[0],
                episodes_per_task=50,
                eval_seed=10000,
                rollout_horizon=600,
                settle_steps=5,
                control_freq=20,
                task_setting="suite_task_id",
                image_shape=[128, 128, 3],
                state_keys=manifest["state_keys"],
                tasks=tasks,
            ),
            indent=2,
        )
        + "\n"
    )
    (output / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    import shutil

    shutil.copyfile(base / "train_history.json", output / "dp_train_history.json")
    shutil.copyfile(
        a.workspace / "outputs/act_spatial_32_parallel/figures/act_audited_aggregate.csv",
        output / "act_learning_curve.csv",
    )
    print(f"Exported {len(exports)} audited episodes across {len(specs)} experiments to {output}")


if __name__ == "__main__":
    main()
