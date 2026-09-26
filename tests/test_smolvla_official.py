"""Exercise native config/API boundaries without loading weights or using CUDA."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("lerobot")

from roboscope.workflows.smolvla_official import (  # noqa: E402
    build_eval_config,
    build_train_config,
    worker_command,
)

ROOT = Path(__file__).resolve().parents[1]


def recipe():
    return json.loads((ROOT / "configs/libero_spatial/smolvla_official.json").read_text())


def test_native_paper_config_roundtrip(tmp_path, monkeypatch):
    from lerobot.configs import policies
    from lerobot.configs.train import TrainPipelineConfig

    cfg = build_train_config(recipe(), tmp_path / "data", tmp_path / "vlm", tmp_path / "train")
    # CPU parsing should preserve the intended device instead of probing CUDA.
    assert cfg.policy.pretrained_path is None
    assert cfg.policy.load_vlm_weights is True
    assert cfg.policy.num_vlm_layers == 16
    assert cfg.policy.expert_width_multiplier == 0.75
    assert (cfg.policy.chunk_size, cfg.policy.n_action_steps, cfg.policy.num_steps) == (50, 1, 10)
    assert cfg.policy.compile_model and cfg.policy.use_amp
    assert (cfg.steps, cfg.batch_size, cfg.dataset.eval_split) == (100000, 64, 0.0)
    assert (cfg.env.observation_height, cfg.env.observation_width) == (256, 256)
    assert cfg.env.task == "libero_spatial"
    assert cfg.env.task_ids is None and cfg.env.episode_length is None
    assert cfg.eval.use_async_envs and cfg.eval.batch_size == recipe()["eval_episodes"]
    assert cfg.env.max_parallel_tasks == 1
    cfg.save_pretrained(tmp_path / "config")
    # Native parsing checks availability; emulate the verified CUDA worker only
    # for that check, without loading any model or initializing CUDA.
    monkeypatch.setattr(policies, "is_torch_device_available", lambda _: True)
    restored = TrainPipelineConfig.from_pretrained(tmp_path / "config")
    assert restored.to_dict() == cfg.to_dict()
    assert restored.policy.get_optimizer_preset().lr == 1e-4
    assert restored.policy.get_scheduler_preset().num_warmup_steps == 1000


def test_smoke_keeps_training_memory_budget_and_limits_evaluation(tmp_path):
    cfg = build_train_config(recipe(), tmp_path, tmp_path, tmp_path / "train", smoke=True)
    assert cfg.batch_size == 64  # A small-batch smoke would miss single-GPU OOMs.
    assert cfg.steps == cfg.save_freq == cfg.env_eval_freq == 2
    assert cfg.env.task_ids == [0] and cfg.env.episode_length == 10
    assert cfg.eval.n_episodes == 1
    assert cfg.eval.batch_size == 1 and not cfg.eval.use_async_envs


def test_native_eval_config_decodes_with_checkpoint(tmp_path, monkeypatch):
    import draccus
    from lerobot.configs import parser
    from lerobot.configs.eval import EvalPipelineConfig
    from lerobot.configs.policies import PreTrainedConfig

    cfg = build_train_config(recipe(), tmp_path, tmp_path, tmp_path / "train")
    checkpoint = tmp_path / "checkpoint"
    monkeypatch.setattr(parser, "get_path_arg", lambda _: str(checkpoint))
    monkeypatch.setattr(parser, "get_cli_overrides", lambda _: [])
    monkeypatch.setattr(parser, "get_yaml_overrides", lambda _: [])
    monkeypatch.setattr(PreTrainedConfig, "from_pretrained", lambda *args, **kwargs: cfg.policy)
    restored = draccus.decode(EvalPipelineConfig, build_eval_config(cfg, recipe(), tmp_path))
    assert restored.policy.pretrained_path == checkpoint
    assert restored.policy.n_action_steps == 1
    assert restored.eval.n_episodes == recipe()["final_eval_episodes"]
    assert restored.env.observation_height == 256


def test_resume_and_evaluate_use_native_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match="resume"):
        worker_command(tmp_path, tmp_path / "configs", tmp_path, "train", resume=True)
    checkpoint = tmp_path / "training/checkpoints/last/pretrained_model"
    checkpoint.mkdir(parents=True)
    (checkpoint / "train_config.json").write_text("{}")
    command = worker_command(tmp_path, tmp_path / "configs", tmp_path, "train", resume=True)
    assert "--resume" in command
    assert command[command.index("--config") + 1] == str(checkpoint / "train_config.json")
    with pytest.raises(FileExistsError, match="resume"):
        worker_command(tmp_path, tmp_path / "configs", tmp_path, "train")
    with pytest.raises(FileNotFoundError, match="evaluate"):
        worker_command(tmp_path, tmp_path / "configs", tmp_path, "evaluate")
    (checkpoint / "model.safetensors").touch()
    command = worker_command(tmp_path, tmp_path / "configs", tmp_path, "evaluate")
    assert command[command.index("--checkpoint") + 1] == str(checkpoint)
    assert command[command.index("--config") + 1] == str(tmp_path / "configs/eval_config.json")


def test_preview_does_not_require_dataset_or_gpu(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "roboscope.workflows.smolvla_official",
            "--recipe",
            str(ROOT / "configs/libero_spatial/smolvla_official.json"),
            "--dataset",
            str(tmp_path / "missing_data"),
            "--cache",
            str(tmp_path / "cache"),
            "--libero-root",
            str(tmp_path / "missing_libero"),
            "--output",
            str(tmp_path / "run"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Preview only" in result.stdout
    assert not (tmp_path / "run").exists()
