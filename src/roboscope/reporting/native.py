"""Read native LeRobot outcomes without inventing initial-state identities."""

import math


def native_episodes(reports, task_ids, episodes_per_task):
    """Validate complete disjoint shards, including their aggregate counters."""
    expected = set(task_ids)
    seen, rows = set(), []
    for report in reports:
        shard = []
        for task in report["per_task"]:
            task_id = task["task_id"]
            if task["task_group"] != "libero_spatial" or task_id not in expected or task_id in seen:
                raise ValueError("Unexpected or duplicate task")
            seen.add(task_id)
            successes = task["metrics"]["successes"]
            if len(successes) != episodes_per_task or any(type(s) is not bool for s in successes):
                raise ValueError("Expected complete boolean episode outcomes")
            shard.extend(
                {"task_id": task_id, "episode_index": index, "success": int(success)}
                for index, success in enumerate(successes)
            )
        overall = report["overall"]
        if not shard or overall["n_episodes"] != len(shard):
            raise ValueError("Aggregate episode count mismatch")
        rate = 100 * sum(r["success"] for r in shard) / len(shard)
        if not math.isclose(rate, overall["pc_success"], abs_tol=1e-8):
            raise ValueError("Aggregate success rate mismatch")
        rows.extend(shard)
    if seen != expected:
        raise ValueError("Missing tasks")
    return sorted(rows, key=lambda r: (r["task_id"], r["episode_index"]))


def audit_snapshot(snapshot):
    """Require the declared coverage and finite curves before plotting."""
    if snapshot["schema"] != 1 or not snapshot["evaluations"]:
        raise ValueError("Unsupported or empty VLA snapshot")
    for study in snapshot["evaluations"]:
        rows = study["episodes"]
        expected = {(t, i) for t in study["task_ids"] for i in range(study["episodes_per_task"])}
        keys = {(r["task_id"], r["episode_index"]) for r in rows}
        if keys != expected or len(rows) != len(expected):
            raise ValueError("Missing or duplicate native episodes")
        if any(type(r["success"]) is not int or r["success"] not in (0, 1) for r in rows):
            raise ValueError("Invalid success")
        if len(study["checkpoint_sha256"]) != 64:
            raise ValueError("Missing checkpoint identity")
    for curve in snapshot["training"].values():
        rows = curve["rows"]
        steps = [r["step"] for r in rows]
        if not steps or steps != sorted(set(steps)) or steps[0] <= 0:
            raise ValueError("Training steps must be unique and increasing")
        if any(not math.isfinite(v) for r in rows for v in r.values() if v is not None):
            raise ValueError("Nonfinite training metric")
    return snapshot
