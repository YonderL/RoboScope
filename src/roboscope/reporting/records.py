"""Audit episode identity before a result can appear in a figure."""

import json
from pathlib import Path


def read_evaluation(directory, episodes_per_task=50, task_ids=None):
    directory = Path(directory)
    task_ids = list(range(10)) if task_ids is None else task_ids
    records, identities = [], []
    for shard_id in range(2):
        shard = directory / f"shard{shard_id}"
        meta = json.loads((shard / "config.json").read_text())
        complete = json.loads((shard / "complete.json").read_text())
        expected = {(t, i) for t in task_ids if t % 2 == shard_id for i in range(episodes_per_task)}
        raw = shard / "episodes.jsonl"
        # A task subset can leave one shard empty; still require its completion
        # marker and checkpoint identity. Missing nonempty shards must fail.
        rows = [json.loads(s) for s in raw.read_text().splitlines()] if raw.exists() or expected else []
        keys = {(r["task_id"], r["initial_state_id"]) for r in rows}
        if len(rows) != len(keys) or keys != expected or complete["episodes"] != len(rows):
            raise ValueError(f"Duplicate, missing, or unexpected episodes: {shard}")
        sha = meta["checkpoint_sha256"]
        if complete["checkpoint_sha256"] != sha or any(r["checkpoint_sha256"] != sha for r in rows):
            raise ValueError("Mixed checkpoint identities")
        for row in rows:
            if (
                row["success"] not in (0, 1)
                or row["eval_seed"]
                != meta["config"]["eval_seed"] + row["task_id"] * 1000 + row["initial_state_id"]
            ):
                raise ValueError("Invalid success or evaluation seed")
        identities.append(
            {"checkpoint_sha256": sha, "gradient_step": meta["gradient_step"], "variant": meta["variant"]}
        )
        records.extend(rows)
    if identities[0] != identities[1]:
        raise ValueError("Shards refer to different models")
    return records, identities[0]


def audit_portable(rows):
    """Published records carry explicit checkpoint labels; never pool best and final."""
    groups = {}
    for r in rows:
        groups.setdefault(r["experiment"], []).append(r)
    for name, group in groups.items():
        keys = {(int(r["task_id"]), int(r["initial_state_id"])) for r in group}
        if len(group) != 500 or len(keys) != 500 or keys != {(t, i) for t in range(10) for i in range(50)}:
            raise ValueError(f"{name}: expected exactly 50 unique initial states for each of 10 tasks")
        if len({r["checkpoint_sha256"] for r in group}) != 1:
            raise ValueError(f"{name}: mixed checkpoints")
        for r in group:
            if int(r["success"]) not in (0, 1) or int(r["eval_seed"]) != 10000 + int(
                r["task_id"]
            ) * 1000 + int(r["initial_state_id"]):
                raise ValueError(f"{name}: invalid outcome/seed")
    return groups
