"""Small CPU regressions: meaningful temporal, conditioning and normalization checks."""

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
from roboscope.data.sequences import SequenceDataset, StepBatchSampler  # noqa: E402
from roboscope.policies.act import Execution  # noqa: E402
from roboscope.policies.diffusion import TaskDiffusionPolicy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return json.loads((ROOT / "configs/libero_spatial/diffusion.json").read_text())


@pytest.fixture
def manifest(tmp_path):
    import h5py

    path = tmp_path / "task.hdf5"
    with h5py.File(path, "w") as f:
        for i in range(2):
            d = f.create_group(f"data/demo_{i}")
            d["actions"] = np.arange(4 * 7, dtype=np.float32).reshape(4, 7) / 30
            d["obs/joint_states"] = np.ones((4, 7), np.float32) * i
            d["obs/gripper_states"] = np.ones((4, 2), np.float32) * i
            for c in ["agentview_rgb", "eye_in_hand_rgb"]:
                d[f"obs/{c}"] = np.full((4, 128, 128, 3), i * 100, np.uint8)
    return {
        "tasks": [
            {
                "id": 0,
                "name": "test",
                "path": str(path),
                "image_convention": "opencv",
                "episodes": [{"demo": f"demo_{i}", "length": 4, "split": "train"} for i in range(2)],
            }
        ],
        "state_mean": [0] * 9,
        "state_std": [1] * 9,
        "action_mean": [0] * 7,
        "action_std": [1] * 7,
        "action_min": [-1] * 7,
        "action_max": [1] * 7,
    }


def test_sequence_never_crosses_episode_boundary(manifest):
    ds = SequenceDataset(manifest, "train")
    assert torch.equal(ds[4]["state"][0], ds[4]["state"][1])
    assert ds[4]["agentview_rgb"].unique().tolist() == [100]
    assert ds[4]["action_is_pad"][0]
    assert not ds[4]["action_is_pad"][1]
    assert torch.equal(ds[3]["action"][1], ds[3]["action"][-1])


def test_sampler_resume_preserves_remaining_batches():
    full = list(StepBatchSampler(97, 16, 20, 0))
    assert full[7:] == list(StepBatchSampler(97, 16, 20, 0, 7))


def test_dp_conditioning_crop_and_action_alignment(cfg, manifest):
    torch.set_num_threads(2)
    cfg.update(down_dims=[16, 32, 64], diffusion_step_embed_dim=16, spatial_keypoints=4, task_embedding_dim=8)
    model = TaskDiffusionPolicy(cfg, manifest)
    ds = SequenceDataset(manifest, "train")
    batch = torch.utils.data.default_collate([ds[0], ds[4]])
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules())
    a, b = list(model.encoders.values())
    assert {p.data_ptr() for p in a.parameters()}.isdisjoint(p.data_ptr() for p in b.parameters())
    model.train()
    loss = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.task_embedding.weight.grad.abs().sum() > 0
    model.eval()
    grid = torch.arange(128 * 128).reshape(1, 1, 128, 128).expand(2, 3, -1, -1)
    assert torch.equal(model.crop_images(grid), grid[:, :, 6:122, 6:122])
    noise = torch.randn(2, 16, 7)
    first = model.predict(batch, 5, noise.clone())
    second = model.predict(batch, 5, noise.clone())
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert first.shape == (2, 15, 7)


def test_temporal_ensemble_aligns_target_times():
    class Fake:
        def __init__(self):
            self.t = 0

        def predict(self, batch):
            x = torch.arange(3).view(1, 3, 1).float() + 10 * self.t
            self.t += 1
            return x

    execution = Execution(Fake(), 3, "ensemble", 0)
    assert [execution.step({}).item() for _ in range(3)] == [0, 5.5, 11]
    execution.reset()
    assert execution.ensemble.ensembled_actions is None
