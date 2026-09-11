import csv
from pathlib import Path

import pytest

from roboscope.reporting.records import audit_portable

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
