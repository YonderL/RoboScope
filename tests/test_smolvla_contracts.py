import json
import subprocess
import sys
from pathlib import Path

import pytest

from roboscope.workflows.config import validate_recipe
from roboscope.workflows.experiments import evaluation_config

ROOT = Path(__file__).resolve().parents[1]


def recipe(name="smolvla"):
    return json.loads((ROOT / f"configs/libero_spatial/{name}.json").read_text())


def test_smolvla_matches_formal_act_dp_protocol_with_native_execution(tmp_path):
    cfg, dp, act = recipe(), recipe("diffusion"), recipe("act_baseline")
    validate_recipe(cfg)
    for key in (
        "eval_episodes",
        "eval_seed",
        "rollout_horizon",
        "settle_steps",
        "video_episodes_per_task",
        "split_seed",
        "validation_fraction",
    ):
        assert cfg[key] == dp[key] == act[key]
    for key in ("eval_envs", "latency_warmup", "latency_repeats", "amp"):
        assert cfg[key] == dp[key]
    assert cfg["eval_envs"] == act["runtime"]["eval_envs"]
    assert cfg["batch_size"] == cfg["micro_batch_size"] == 64 and cfg["train_steps"] == 20000
    assert cfg["chunk_size"] == 50 and cfg["action_horizon"] == 10
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {"tasks": [{"id": i, "language": f"task {i}", "eval_initial_state_ids": [0]} for i in range(10)]}
        )
    )
    effective, manifest, variant = evaluation_config(tmp_path, "final", 50, 10, None)
    assert effective["policy"] == "smolvla" and effective["action_horizon"] == 10
    assert variant == "smolvla_chunk10"
    assert all(t["eval_initial_state_ids"] == list(range(50)) for t in manifest["tasks"])
    with pytest.raises(ValueError, match="action horizon"):
        evaluation_config(tmp_path, "final", 50, 10, 8)
    cfg["policy"] = "diffusion"
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert evaluation_config(tmp_path, "final", 50, 10, None)[0]["action_horizon"] == 8


def test_smolvla_preview_needs_no_data_or_weights(tmp_path):
    output = tmp_path / "untouched"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "roboscope",
            "train",
            "--recipe",
            str(ROOT / "configs/libero_spatial/smolvla.json"),
            "--data-root",
            "/missing",
            "--libero-root",
            "/missing",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Preview only" in result.stdout
    assert not output.exists()
