import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from roboscope.workflows.config import validate_rlt
from roboscope.workflows.experiments import evaluation_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("stage", ["token", "warmup", "online", "evaluate", "all"])
def test_stage_script_preview_dispatches_without_loading_assets(tmp_path, stage):
    output = tmp_path / "no_run"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/posttrain_smolvla_rlt_spatial.sh"),
            "--stage",
            stage,
            "--source",
            "/missing/sft",
            "--output",
            str(output),
            "--preview",
        ],
        env={**os.environ, "PYTHON": sys.executable},
        capture_output=True,
        text=True,
        check=True,
    )
    assert not output.exists()
    assert result.stdout.count('"command": "evaluate"') == (0 if stage in ("token", "warmup") else 2)
    assert ('"recipe"' in result.stdout) == (stage != "evaluate")
    if stage != "evaluate":
        assert f'"stage": "{stage}"' in result.stdout


def test_rlt_preview_and_eval_protocol(tmp_path):
    recipe = ROOT / "configs/libero_spatial/smolvla_rlt.json"
    cfg = json.loads(recipe.read_text())
    validate_rlt(cfg)
    sft = json.loads((ROOT / "configs/libero_spatial/smolvla.json").read_text())
    for key in (
        "eval_episodes",
        "eval_seed",
        "rollout_horizon",
        "settle_steps",
        "eval_envs",
        "latency_warmup",
        "latency_repeats",
        "video_episodes_per_task",
    ):
        assert cfg[key] == sft[key]
    assert cfg["action_horizon"] == sft["action_horizon"] == 10
    output = tmp_path / "untouched"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "roboscope",
            "posttrain",
            "--recipe",
            str(recipe),
            "--source",
            "/missing/sft",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert "No checkpoint loaded" in result.stdout and not output.exists()
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "manifest.json").write_text(json.dumps({"tasks": [{"id": i} for i in range(10)]}))
    trained, manifest, name = evaluation_config(tmp_path, "final", 50, 10, None)
    reference, _, ref_name = evaluation_config(tmp_path, "final", 50, 10, None, True)
    assert trained["action_horizon"] == reference["action_horizon"] == 10
    assert name == "smolvla_rlt_c010" and ref_name == "smolvla_reference_c010"
    assert reference["evaluation_policy"] == "sft_reference"
    assert all(t["eval_initial_state_ids"] == list(range(50)) for t in manifest["tasks"])
    with pytest.raises(ValueError, match="trained action horizon"):
        evaluation_config(tmp_path, "final", 50, 10, 50)
    with pytest.raises(ValueError, match="final"):
        evaluation_config(tmp_path, "best", 50, 10, None)


def test_sft_reference_single_step_execution_preserves_rlt_contract(tmp_path):
    cfg = json.loads((ROOT / "configs/libero_spatial/smolvla_rlt_hf.json").read_text())
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "manifest.json").write_text(json.dumps({"tasks": [{"id": i} for i in range(10)]}))
    single, schedule, name = evaluation_config(tmp_path, "final", 50, 10, 1, True)
    assert single["evaluation_policy"] == "sft_reference"
    assert single["hf_eval_reset_protocol"] == "property_sampler_clear_v1"
    assert single["action_horizon"] == 1 and name == "smolvla_reference_c001"
    assert all(t["eval_initial_state_ids"] == list(range(50)) for t in schedule["tasks"])
    assert json.loads((tmp_path / "config.json").read_text())["action_horizon"] == 10
    with pytest.raises(ValueError, match="trained action horizon"):
        evaluation_config(tmp_path, "final", 50, 10, 1)
    with pytest.raises(ValueError, match="execution horizon"):
        evaluation_config(tmp_path, "final", 50, 10, 50, True)
    with pytest.raises(ValueError, match="DDIM"):
        evaluation_config(tmp_path, "final", 50, 5, 1, True)


@pytest.mark.parametrize(
    "change",
    [
        {"gamma": 1.1},
        {"token_dim": 255},
        {"warmup_steps": 100000},
        {"replay_stride": 11},
        {"replay_capacity": 1},
        {"actor_std": 0},
        {"token_features": "full_prefix"},
    ],
)
def test_invalid_rlt_recipe_rejected(change):
    cfg = json.loads((ROOT / "configs/libero_spatial/smolvla_rlt.json").read_text())
    with pytest.raises(ValueError):
        validate_rlt({**cfg, **change})
