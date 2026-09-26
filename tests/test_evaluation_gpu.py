"""Explicit evaluation GPU routing must never silently use the other model."""

import json

import pytest

pytest.importorskip("torch")
from roboscope.runtime import common, gpu  # noqa: E402
from roboscope.workflows import experiments  # noqa: E402


def test_gpu_inventory_and_worker_require_explicit_model(monkeypatch):
    inventory = (
        "0, NVIDIA RTX 5880 Ada Generation, GPU-a, 00000000:16:00.0\n"
        "1, NVIDIA RTX 5880 Ada Generation, GPU-b, 00000000:38:00.0\n"
        "2, NVIDIA GeForce RTX 4090, GPU-c, 00000000:C8:00.0\n"
        "3, NVIDIA GeForce RTX 4090, GPU-d, 00000000:D8:00.0\n"
    )
    monkeypatch.setattr(gpu.subprocess, "check_output", lambda *a, **kw: inventory)
    assert [c["index"] for c in gpu.gpu_inventory("5880")] == [0, 1]
    assert [c["index"] for c in gpu.gpu_inventory()] == [2, 3]
    monkeypatch.setattr(common.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(common.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(common.torch.cuda, "get_device_name", lambda _: "NVIDIA RTX 5880 Ada Generation")
    common.require_gpu("5880")
    with pytest.raises(RuntimeError, match="non-4090"):
        common.require_4090()
    inventory = inventory.splitlines()[0]
    with pytest.raises(RuntimeError, match="exactly two 5880"):
        gpu.gpu_inventory("5880")


@pytest.mark.parametrize("gpu_model", [None, "5880", "pro5000"])
def test_evaluation_explicit_gpus_parallel_default_smolvla_serial(tmp_path, monkeypatch, gpu_model):
    from roboscope.reporting import records

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(json.dumps({"policy": "smolvla", "action_horizon": 10}))
    (source / "manifest.json").write_text(json.dumps({"tasks": []}))
    (source / "best.pt").touch()
    calls, selected, configs = [], [], []
    envs = [{"CUDA_VISIBLE_DEVICES": "GPU-a"}, {"CUDA_VISIBLE_DEVICES": "GPU-b"}]

    def mapping(model):
        selected.append(model)
        return [], envs

    monkeypatch.setattr(gpu, "cards_and_envs", mapping)
    monkeypatch.setattr(gpu, "run_workers", lambda commands, envs, logs: calls.append((commands, envs)))
    monkeypatch.setattr(experiments, "save_contract", lambda run, cfg, *args: configs.append(cfg.copy()))
    monkeypatch.setattr(records, "read_evaluation", lambda *args: ([{"success": 1}], {}))
    experiments.evaluate(source, tmp_path / "eval", checkpoint="best", gpu_model=gpu_model)
    assert selected == [gpu_model or "4090"]
    assert configs[0]["action_horizon"] == 10
    assert configs[0]["eval_episodes"] == 50
    if gpu_model:
        assert len(calls) == 1 and calls[0][1] == envs
        assert [cmd[-1] for cmd in calls[0][0]] == ["0", "1"]
        assert configs[0]["evaluation_gpu_model"] == gpu_model
    else:
        assert len(calls) == 2 and all(env == envs[-1:] for _, env in calls)
