"""The official selection must not add reruns to the old trial denominator."""

import csv
import json
import re
import shutil
from pathlib import Path

import pytest

from roboscope.reporting.official import build_summary

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "act": [42, 48, 45, 48, 38, 41, 46, 34, 43, 49],
    "dp": [48, 50, 45, 47, 35, 49, 47, 37, 44, 42],
    "smolvla": [47, 49, 44, 49, 38, 47, 45, 47, 42, 40],
    "pi0": [50, 49, 46, 47, 37, 42, 45, 42, 38, 42],
}


def test_official_counts_task_alignment_and_single_replacement():
    summary = build_summary(ROOT / "results")
    for policy, expected in EXPECTED.items():
        rows = [r for r in summary["per_task"] if r["policy"] == policy]
        assert [r["successes"] for r in rows] == expected
        assert summary["totals"][policy]["successes"] == sum(expected)
        assert sum(r["episodes"] for r in rows) == 500
        if policy != "pi0":
            assert [r["task_id"] for r in rows if r["source"].startswith("mujoco332/")] == [7]
    ramekin = next(r for r in summary["per_task"] if r["policy"] == "smolvla" and r["task_id"] == 7)
    assert ramekin["source_task_id"] == 5
    assert ramekin["identity_kind"] == "native_episode_index"
    assert "on_the_ramekin" in ramekin["task_name"]


def test_published_official_table_is_reproducible():
    published = json.loads((ROOT / "results/official_spatial/summary.json").read_text())
    assert build_summary(ROOT / "results") == published


@pytest.mark.parametrize("readme", ["README.md", "README.zh-CN.md"])
def test_readme_suite_totals_match_official_selection(readme):
    text = (ROOT / readme).read_text()
    summary = json.loads((ROOT / "results/official_spatial/summary.json").read_text())
    for label, policy in (
        ("ACT", "act"),
        ("Diffusion Policy", "dp"),
        ("SmolVLA", "smolvla"),
        ("Pi-0 LoRA", "pi0"),
    ):
        total = summary["totals"][policy]
        row = (
            rf"\|\s*{re.escape(label)}[^|\n]*\|\s*{total['successes']}/500\s*\|"
            rf"\s*\*\*{total['success_rate_percent']:.1f}%\*\*"
        )
        assert re.search(row, text), (readme, label)


@pytest.mark.parametrize("damage", ["duplicate", "wrong_task_name", "wrong_checkpoint"])
def test_corrupt_replacement_cannot_be_published(tmp_path, damage):
    results = tmp_path / "results"
    for directory in ("libero_spatial", "vla_spatial", "mujoco332"):
        shutil.copytree(ROOT / "results" / directory, results / directory)
    path = results / "mujoco332/episodes.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if damage == "duplicate":
        rows.append(rows[0])
    elif damage == "wrong_task_name":
        rows[0]["task_name"] = "next_to_ramekin_is_not_on_ramekin"
    else:
        rows[0]["checkpoint_sha256"] = "wrong"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError):
        build_summary(results)
