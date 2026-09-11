import json
from pathlib import Path

import pytest

from roboscope.workflows.config import validate_recipe
from roboscope.workflows.experiments import evaluation_config, snapshot

ROOT = Path(__file__).resolve().parents[1]


def test_horizon_cannot_silently_differ_from_implemented_window():
    cfg = json.loads((ROOT / "configs/libero_spatial/diffusion.json").read_text())
    cfg["prediction_horizon"] = 32
    with pytest.raises(ValueError, match="prediction_horizon"):
        validate_recipe(cfg)


def test_source_resume_refuses_changed_code(tmp_path):
    snapshot(tmp_path)
    archived = tmp_path / "source/roboscope/cli.py"
    archived.write_text(archived.read_text() + "\n# unexpected edit\n")
    with pytest.raises(ValueError, match="Source changed"):
        snapshot(tmp_path)


def test_legacy_dp_paths_and_new_initial_state_schedule(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"seed": 0, "policy": "diffusion"}))
    (tmp_path / "env_config.json").write_text(
        json.dumps({"data_root": "/fixture/data", "libero_root": "/fixture/assets"})
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"tasks": [{"eval_initial_state_ids": [0]}]}))
    cfg, manifest, variant = evaluation_config(tmp_path, "final", 50, 10, 8)
    assert cfg["libero_root"] == "/fixture/assets"
    assert manifest["tasks"][0]["eval_initial_state_ids"] == list(range(50))
    assert variant == "dp_ddim10_ta08"
