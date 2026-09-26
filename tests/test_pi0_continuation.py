"""Migration preserves learned state and sampling budget without weakening resume checks."""

import copy
import json

import pytest

torch = pytest.importorskip("torch")
from test_pi0_training import TinyDataset, TinyPolicy, config  # noqa: E402

from roboscope.runtime.gpu import gpu_inventory  # noqa: E402
from roboscope.trainers import pi0  # noqa: E402
from roboscope.workflows.pi0_continue import continuation_config, prepare_continuation  # noqa: E402


def test_continuation_changes_only_execution_partition(tmp_path):
    cfg = {**config(2), "policy": "pi0_lora", "output_root": "parent"}
    old = {"config": cfg, "step": 2, "rng_by_rank": [{}, {}], "optimizer": {}}
    before = copy.deepcopy(old)
    new = continuation_config(old, tmp_path / "parent", tmp_path / "child", "abc", 4)
    assert old == before
    assert new["micro_batch_size"] == 4 and new["gradient_accumulation_steps"] == 1
    assert new["validation_micro_batch_size"] == cfg["micro_batch_size"]
    assert new["batch_size"] == cfg["batch_size"]
    for key in ("lr", "warmup_steps", "train_steps", "seed", "weight_decay", "grad_clip"):
        assert new[key] == cfg[key]
    assert new["continuation"]["parent_config"] == cfg
    for rank in range(2):

        def samples(c):
            sampler = pi0.StepMicroBatchSampler(
                13,
                c["micro_batch_size"],
                c["gradient_accumulation_steps"],
                4,
                c["seed"],
                rank,
                2,
                start_step=2,
            )
            return [index for batch in sampler for index in batch]

        assert samples(new) == samples(cfg)
    with pytest.raises(ValueError, match="divide"):
        continuation_config(old, tmp_path, tmp_path, "abc", 3)


def test_short_run_retains_full_lr_schedule_and_resume(tmp_path):
    cfg, data, device = config(), TinyDataset(), torch.device("cpu")
    pi0.train(tmp_path, cfg, {}, TinyPolicy(), data, data, device, stop_after=1)
    first = torch.load(tmp_path / "last.pt", weights_only=False)
    assert first["step"] == 1 and first["config"]["train_steps"] == 4
    assert not (tmp_path / "final.pt").exists()
    assert first["history"][-1]["lr"] == pi0.learning_rate(0, cfg)
    pi0.train(tmp_path, cfg, {}, TinyPolicy(), data, data, device, resume=True)
    final = torch.load(tmp_path / "last.pt", weights_only=False)
    assert final["step"] == 4
    assert final["history"][-1]["lr"] == pi0.learning_rate(3, cfg)


def test_pro5000_selection_survives_gpu_renumbering(monkeypatch):
    lines = [
        "0, NVIDIA RTX 5880 Ada Generation, GPU-a, 00000000:16:00.0",
        "1, NVIDIA RTX 5880 Ada Generation, GPU-b, 00000000:38:00.0",
        "2, NVIDIA RTX PRO 5000 72GB Blackwell, GPU-c, 00000000:42:00.0",
        "3, NVIDIA RTX PRO 5000 72GB Blackwell, GPU-d, 00000000:43:00.0",
        "4, NVIDIA GeForce RTX 4090, GPU-e, 00000000:C8:00.0",
        "5, NVIDIA GeForce RTX 4090, GPU-f, 00000000:D8:00.0",
    ]
    monkeypatch.setattr("roboscope.runtime.gpu.subprocess.check_output", lambda *a, **kw: "\n".join(lines))
    assert [c["uuid"] for c in gpu_inventory("pro5000")] == ["GPU-c", "GPU-d"]
    assert [c["index"] for c in gpu_inventory()] == [4, 5]


def test_migration_preserves_parent_and_optimizer(tmp_path, monkeypatch):
    parent, child = tmp_path / "parent", tmp_path / "child"
    parent.mkdir()
    child.mkdir()
    cfg = {**config(2), "policy": "pi0_lora", "output_root": str(parent)}
    manifest = {"data": "unchanged"}
    checkpoint = {
        "config": cfg,
        "manifest": manifest,
        "manifest_sha256": pi0.manifest_digest(manifest),
        "step": 2,
        "adapter": {"weight": torch.tensor([1.25])},
        "optimizer": {"state": {0: {"exp_avg": torch.tensor([0.2]), "step": torch.tensor(2)}}},
        "rng_by_rank": [{"seed": 1}, {"seed": 2}],
        "history": [{"step": 2}],
    }
    torch.save(checkpoint, parent / "last.pt")
    torch.save(checkpoint, parent / "best.pt")
    (parent / "manifest.json").write_text(json.dumps(manifest))
    (parent / "image_cache").mkdir()
    original_bytes = (parent / "last.pt").read_bytes()
    monkeypatch.setattr("roboscope.workflows.experiments.save_contract", lambda *args, **kwargs: None)
    new_cfg = prepare_continuation(parent, child, 4)
    migrated = torch.load(child / "last.pt", weights_only=False)
    assert (parent / "last.pt").read_bytes() == original_bytes
    assert migrated["config"] == new_cfg
    assert migrated["rng_by_rank"] == checkpoint["rng_by_rank"]
    assert migrated["history"] == checkpoint["history"]
    torch.testing.assert_close(migrated["optimizer"]["state"][0]["exp_avg"], torch.tensor([0.2]))
    torch.testing.assert_close(migrated["adapter"]["weight"], torch.tensor([1.25]))
    assert (child / "image_cache").resolve() == parent / "image_cache"
    assert torch.load(child / "best.pt", weights_only=False)["config"] == new_cfg
