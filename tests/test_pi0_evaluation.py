"""CPU-only checks for evaluation integrity and aggregation."""

import json

import pytest

from roboscope.evaluation.pi0 import (
    aggregate_evaluation,
    append_record,
    read_records,
    summarize_records,
    wilson_interval,
)


def episode(task, initial, success=0):
    return {
        "task_id": task,
        "task_name": f"task_{task}",
        "initial_state_id": initial,
        "success": success,
        "steps": 220,
        "checkpoint_sha256": "weights",
        "model_calls": 28,
        "clipped_steps": 2,
        "rollout_seconds": 12,
        "inference_samples_ms": [10, 20, 30],
    }


def test_wilson_interval_extremes_and_symmetry():
    assert wilson_interval(0, 50)[0] == 0
    assert wilson_interval(50, 50)[1] == 1
    low, high = wilson_interval(25, 50)
    assert low == pytest.approx(1 - high)
    assert low < 0.5 < high
    with pytest.raises(ValueError):
        wilson_interval(0, 0)


def test_summary_includes_failures_in_denominator_and_task_metrics():
    rows = [episode(task, initial, int(task == 0)) for task in range(2) for initial in range(2)]
    summary = summarize_records(rows, [0, 1], 2, "weights")
    assert summary["episodes"] == 4
    assert summary["success_rate"] == 0.5
    assert summary["macro_task_success_rate"] == 0.5
    assert [task["success_rate"] for task in summary["per_task"]] == [1, 0]
    assert summary["rollout_inference_p50_ms"] == 20
    assert summary["clipped_step_fraction"] == pytest.approx(2 / 220)


@pytest.mark.parametrize(
    "rows,match",
    [
        ([], "Incomplete"),
        ([episode(0, 0), episode(0, 0)], "Duplicate"),
        ([episode(1, 0)], "unexpected"),
        ([{**episode(0, 0), "checkpoint_sha256": "different"}], "checkpoint"),
        ([{**episode(0, 0), "success": None}], "binary"),
        ([{**episode(0, 0), "error": "EGL initialization failed"}], "errors"),
        ([{**episode(0, 0), "steps": 221}], "horizon"),
    ],
)
def test_summary_rejects_missing_duplicate_corrupt_or_error_trials(rows, match):
    with pytest.raises(ValueError, match=match):
        summarize_records(rows, [0], 1, "weights")


def test_resume_recovers_only_unterminated_final_record(tmp_path):
    path = tmp_path / "episodes.jsonl"
    append_record(path, episode(0, 0))
    with path.open("ab") as handle:
        handle.write(b'{"task_id":')
    assert read_records(path, repair_tail=True) == [episode(0, 0)]
    append_record(path, episode(0, 1))
    assert len(read_records(path)) == 2
    with path.open("ab") as handle:
        handle.write(b"broken\n")
    with pytest.raises(ValueError, match="Corrupt"):
        read_records(path, repair_tail=True)


def test_resume_preserves_complete_record_without_newline(tmp_path):
    path = tmp_path / "episodes.jsonl"
    path.write_text(json.dumps(episode(0, 0)))
    assert len(read_records(path, repair_tail=True)) == 1
    append_record(path, episode(0, 1))
    assert len(read_records(path)) == 2


def create_shards(output):
    identity = {
        "checkpoint_sha256": "weights",
        "checkpoint": "/fixture/final.pt",
        "episodes_per_task": 2,
        "task_ids": [0, 1],
        "smoke": True,
        "protocol": {"rollout_horizon": 220},
    }
    for shard in range(2):
        folder = output / f"shard_{shard}"
        folder.mkdir()
        (folder / "metadata.json").write_text(
            json.dumps({**identity, "shard": shard, "shard_task_ids": [shard]})
        )
        for initial in range(2):
            append_record(folder / "episodes.jsonl", episode(shard, initial, int(shard == 0)))
        (folder / "complete.json").write_text(json.dumps({"episodes": 2, "checkpoint_sha256": "weights"}))


def test_two_shards_aggregate_without_double_counting(tmp_path):
    create_shards(tmp_path)
    summary = aggregate_evaluation(tmp_path)
    assert summary["episodes"] == 4 and summary["success_rate"] == 0.5
    assert summary["smoke"] is True
    assert (tmp_path / "episodes.csv").read_text().count("\n") == 5
    assert (tmp_path / "task_metrics.csv").read_text().count("\n") == 3
    assert json.loads((tmp_path / "summary.json").read_text())["checkpoint_sha256"] == "weights"


def test_aggregate_requires_both_shards_and_same_protocol(tmp_path):
    create_shards(tmp_path)
    path = tmp_path / "shard_1/metadata.json"
    value = json.loads(path.read_text())
    value["protocol"]["rollout_horizon"] = 600
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Mixed evaluation"):
        aggregate_evaluation(tmp_path)
    (tmp_path / "shard_1/complete.json").unlink()
    with pytest.raises(FileNotFoundError):
        aggregate_evaluation(tmp_path)
