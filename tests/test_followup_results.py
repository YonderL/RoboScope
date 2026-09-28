"""Different episode conventions cannot silently be merged or double-counted."""

import csv
import json

import pytest

from roboscope.reporting.evaluation import load_results


def write_fixture(root, rows):
    (root / "protocol.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "studies": {"act": {"task_ids": [7], "episodes_per_task": 1, "checkpoint_sha256": "weights"}},
            }
        )
    )
    with (root / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def episode():
    return {
        "policy": "act",
        "task_id": 7,
        "trial_index": 0,
        "identity_kind": "fixed_initial_state",
        "eval_seed": 17000,
        "success": 1,
        "checkpoint_sha256": "weights",
    }


def test_complete_fixed_state_trial(tmp_path):
    write_fixture(tmp_path, [episode()])
    _, groups = load_results(tmp_path)
    assert len(groups["act"]) == 1


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("identity_kind", "native_episode_index", "identity"),
        ("eval_seed", 0, "seed"),
        ("checkpoint_sha256", "different", "checkpoint"),
        ("success", "false", "outcome"),
    ],
)
def test_reject_incompatible_identity_or_outcome(tmp_path, field, value, message):
    write_fixture(tmp_path, [{**episode(), field: value}])
    with pytest.raises(ValueError, match=message):
        load_results(tmp_path)


def test_duplicate_trial_cannot_inflate_score(tmp_path):
    write_fixture(tmp_path, [episode(), episode()])
    with pytest.raises(ValueError, match="duplicate"):
        load_results(tmp_path)
