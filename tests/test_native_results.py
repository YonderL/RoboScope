"""Result corruption must fail before a native VLA score is published."""

import copy
import json
from pathlib import Path

import pytest

from roboscope.reporting.native import audit_snapshot, native_episodes


def report():
    return {
        "per_task": [{"task_group": "libero_spatial", "task_id": 5, "metrics": {"successes": [True, False]}}],
        "overall": {"n_episodes": 2, "pc_success": 50.0},
    }


def test_native_coverage_and_aggregate_are_both_checked():
    value = report()
    assert [r["success"] for r in native_episodes([value], [5], 2)] == [1, 0]
    with pytest.raises(ValueError, match="duplicate"):
        native_episodes([value, value], [5], 2)
    with pytest.raises(ValueError, match="Missing"):
        native_episodes([value], [5, 6], 2)
    value["overall"]["pc_success"] = 100
    with pytest.raises(ValueError, match="success rate"):
        native_episodes([value], [5], 2)


def test_native_outcome_is_not_truthiness_coerced():
    value = report()
    value["per_task"][0]["metrics"]["successes"][1] = "False"
    with pytest.raises(ValueError, match="boolean"):
        native_episodes([value], [5], 2)


def test_published_vla_snapshot_and_duplicate_rejection():
    path = Path(__file__).resolve().parents[1] / "results/vla_spatial/snapshot.json"
    value = audit_snapshot(json.loads(path.read_text()))
    full = value["evaluations"][1]
    assert len(full["episodes"]) == 500
    assert sum(r["success"] for r in full["episodes"]) == 411
    damaged = copy.deepcopy(value)
    damaged["evaluations"][1]["episodes"][0] = damaged["evaluations"][1]["episodes"][1]
    with pytest.raises(ValueError, match="duplicate"):
        audit_snapshot(damaged)
