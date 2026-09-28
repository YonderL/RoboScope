import csv
import json
from pathlib import Path

import pytest

from roboscope.reporting.records import audit_portable, read_evaluation

ROOT = Path(__file__).resolve().parents[1]


def records():
    with (ROOT / "results/libero_spatial/episodes.csv").open() as h:
        return list(csv.DictReader(h))


def test_published_counts_and_pairing():
    groups = audit_portable(records())
    assert len(groups) == 7
    assert sum(int(r["success"]) for r in groups["act_final"]) == 415
    assert sum(int(r["success"]) for r in groups["dp_best"]) == 347
    assert sum(int(r["success"]) for r in groups["dp_final"]) == 409


def test_duplicate_cannot_inflate_sr():
    rows = records()
    rows[1] = dict(rows[0])
    with pytest.raises(ValueError, match="unique"):
        audit_portable(rows)


def test_mixed_checkpoint_is_rejected():
    rows = records()
    rows[0]["checkpoint_sha256"] = "different-checkpoint"
    with pytest.raises(ValueError, match="mixed"):
        audit_portable(rows)


def test_incomplete_results_are_not_reported():
    with pytest.raises(ValueError, match="unique"):
        audit_portable(records()[:-1])


def test_task_subset_allows_empty_shard_but_requires_nonempty_records(tmp_path):
    for shard in range(2):
        path = tmp_path / f"shard{shard}"
        path.mkdir()
        (path / "config.json").write_text(
            json.dumps(
                {
                    "checkpoint_sha256": "weights",
                    "gradient_step": 30,
                    "variant": "dp",
                    "config": {"eval_seed": 10000},
                }
            )
        )
        (path / "complete.json").write_text(json.dumps({"episodes": shard, "checkpoint_sha256": "weights"}))
    episode = {
        "task_id": 7,
        "initial_state_id": 0,
        "eval_seed": 17000,
        "success": 1,
        "checkpoint_sha256": "weights",
    }
    raw = tmp_path / "shard1/episodes.jsonl"
    raw.write_text(json.dumps(episode) + "\n")
    rows, _ = read_evaluation(tmp_path, 1, [7])
    assert rows == [episode]
    raw.unlink()
    with pytest.raises(FileNotFoundError):
        read_evaluation(tmp_path, 1, [7])
