"""Real optimizer, restart and two-rank tests using a tiny policy, never base weights."""

import json
import os
from datetime import timedelta
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from roboscope.policies.pi0 import LoRALinear  # noqa: E402
from roboscope.trainers import pi0  # noqa: E402


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 13

    def __getitem__(self, index):
        return {"state": torch.tensor([index / 13, 0.5]), "target": torch.tensor([0.25])}


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(74)
            self.layer = LoRALinear(nn.Linear(2, 1), rank=2, alpha=2)
        # Real Pi-0 has final VLM branch parameters unused by its suffix loss.
        self.unused = nn.Parameter(torch.ones(2))

    def forward(self, batch):
        noise = torch.randn_like(batch["target"]) * 0.01
        return (self.layer(batch["state"]) - batch["target"] - noise).square().mean()

    def trainable_state_dict(self):
        return {k: p.detach().cpu().clone() for k, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state_dict(self, values):
        with torch.no_grad():
            for name, value in values.items():
                self.get_parameter(name).copy_(value)


def config(world=1):
    return dict(
        world_size=world,
        batch_size=4 * world,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        seed=3,
        lr=0.01,
        weight_decay=0.01,
        warmup_steps=1,
        min_lr_ratio=0.1,
        train_steps=4,
        workers=0,
        validate_every=2,
        save_every=1,
        validation_samples=5,
        log_every=1,
        grad_clip=1.0,
    )


def test_sampler_exact_resume_across_epoch_boundary():
    kwargs = dict(size=13, micro_batch_size=2, accumulation_steps=2, steps=5, seed=2026, world_size=2)
    ranks = [list(pi0.StepMicroBatchSampler(**kwargs, rank=rank)) for rank in range(2)]
    for rank in range(2):
        resumed = list(pi0.StepMicroBatchSampler(**kwargs, rank=rank, start_step=2))
        assert resumed == ranks[rank][4:]
    first_epoch = []
    for step in range(2):
        for rank in range(2):
            first_epoch.extend(index for batch in ranks[rank][step * 2 : (step + 1) * 2] for index in batch)
    assert len(set(first_epoch[:13])) == 13


def test_gradient_accumulation_is_batch_mean():
    class Linear(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(0.2))

        def forward(self, batch):
            return (batch["state"] * self.weight).square().mean()

    micro = Linear()
    batches = [{"state": torch.tensor([1.0, 2.0])}, {"state": torch.tensor([3.0, 4.0])}]
    pi0.accumulate_gradients(micro, iter(batches), 2, torch.device("cpu"))
    reference = Linear()
    reference({"state": torch.arange(1.0, 5.0)}).backward()
    torch.testing.assert_close(micro.weight.grad, reference.weight.grad)


def test_restart_matches_uninterrupted_training(tmp_path, monkeypatch):
    cfg, manifest = config(), {"pi0_action_mean": [0] * 7}
    device, dataset = torch.device("cpu"), TinyDataset()
    full = tmp_path / "full"
    pi0.train(full, cfg, manifest, TinyPolicy(), dataset, dataset, device)
    interrupted = tmp_path / "interrupted"
    atomic = pi0.atomic_checkpoint

    class StopAfterCheckpoint(Exception):
        pass

    def save_then_stop(path, payload):
        atomic(path, payload)
        if Path(path).name == "last.pt" and payload["step"] == 2:
            raise StopAfterCheckpoint

    monkeypatch.setattr(pi0, "atomic_checkpoint", save_then_stop)
    with pytest.raises(StopAfterCheckpoint):
        pi0.train(interrupted, cfg, manifest, TinyPolicy(), dataset, dataset, device)
    monkeypatch.setattr(pi0, "atomic_checkpoint", atomic)
    pi0.train(interrupted, cfg, manifest, TinyPolicy(), dataset, dataset, device, resume=True)
    expected = torch.load(full / "final.pt", weights_only=False)
    actual = torch.load(interrupted / "final.pt", weights_only=False)
    for key in expected["adapter"]:
        torch.testing.assert_close(actual["adapter"][key], expected["adapter"][key], rtol=0, atol=0)
    assert actual["validation_loss"] == expected["validation_loss"]
    assert len((interrupted / "train_metrics.jsonl").read_text().splitlines()) == cfg["train_steps"]
    with pytest.raises(ValueError, match="config differs"):
        pi0.resume_checkpoint(interrupted, {**cfg, "lr": 0.1}, manifest, 1, True)
    with pytest.raises(ValueError, match="manifest differs"):
        pi0.resume_checkpoint(interrupted, cfg, {}, 1, True)


def distributed_worker(rank, directory, rendezvous, cuda):
    torch.set_num_threads(1)
    device = torch.device("cuda", rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        "nccl" if cuda else "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        model = TinyPolicy().to(device)
        cfg = config(2)
        pi0.train(directory, cfg, {}, model, TinyDataset(), TinyDataset(), device, rank, 2)
        values = torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])
        gathered = [torch.empty_like(values) for _ in range(2)]
        torch.distributed.all_gather(gathered, values)
        torch.testing.assert_close(gathered[0], gathered[1], rtol=0, atol=0)
        # Re-enter the restore path on both ranks and verify completion is idempotent.
        restored = TinyPolicy().to(device)
        pi0.train(directory, cfg, {}, restored, TinyDataset(), TinyDataset(), device, rank, 2, resume=True)
        for key, value in restored.trainable_state_dict().items():
            torch.testing.assert_close(value, model.trainable_state_dict()[key], rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


def test_two_rank_training_checkpoint(tmp_path):
    cuda = os.environ.get("ROBOSCOPE_TEST_CUDA_DDP") == "1"
    if cuda:
        assert torch.cuda.device_count() == 2
        assert all("4090" in torch.cuda.get_device_name(i) for i in range(2))
    run = tmp_path / "ddp"
    torch.multiprocessing.spawn(
        distributed_worker,
        args=(str(run), str(tmp_path / "rendezvous"), cuda),
        nprocs=2,
        join=True,
    )
    saved = torch.load(run / "last.pt", weights_only=False)
    assert len(saved["rng_by_rank"]) == 2
    assert saved["step"] == 4
    assert not any("base" in name for name in saved["adapter"])
    history = json.loads((run / "train_history.json").read_text())
    assert history[-1]["samples_seen"] == 32
    assert "gpu_1_peak_allocated_gib" in history[-1]
