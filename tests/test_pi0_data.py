import json
from pathlib import Path

import numpy as np
import pytest

from roboscope.workflows.config import validate_recipe

ROOT = Path(__file__).resolve().parents[1]


def test_pi0_batch_contract():
    cfg = json.loads((ROOT / "configs/libero_spatial/pi0_lora.json").read_text())
    validate_recipe(cfg)
    cfg["gradient_accumulation_steps"] = 8
    with pytest.raises(ValueError, match="batch_size"):
        validate_recipe(cfg)


def test_pi0_preview_without_training_dependencies(tmp_path):
    import subprocess
    import sys

    output = tmp_path / "untouched"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "roboscope",
            "train",
            "--recipe",
            str(ROOT / "configs/libero_spatial/pi0_lora.json"),
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


def test_pi0_image_state_and_terminal_actions(tmp_path):
    h5py = pytest.importorskip("h5py")
    torch = pytest.importorskip("torch")
    from roboscope.data.pi0 import Pi0Dataset, observation_batch

    path = tmp_path / "demo.hdf5"
    images = np.zeros((2, 128, 128, 3), dtype=np.uint8)
    images[:, 0, :, :] = 255
    state = np.array([0.1, 0.2, 0.3, 0, 0, np.pi / 2, 0.02, 0.03], dtype=np.float32)
    with h5py.File(path, "w") as f:
        demo = f.create_group("data/demo_0")
        demo["actions"] = np.array([[0] * 7, [0.5] * 7], dtype=np.float32)
        obs = demo.create_group("obs")
        obs["joint_states"] = np.zeros((2, 7), dtype=np.float32)
        obs["gripper_states"] = np.tile(state[6:], (2, 1))
        obs["ee_states"] = np.tile(state[:6], (2, 1))
        for key in ("agentview_rgb", "eye_in_hand_rgb"):
            obs[key] = images
    task = {
        "id": 0,
        "path": str(path),
        "language": "pick up the bowl",
        "image_convention": "opengl",
        "episodes": [{"demo": "demo_0", "length": 2, "split": "train"}],
    }
    dataset = Pi0Dataset({"tasks": [task]}, 50, "train")
    row = dataset[1]
    assert row["action"].shape == (50, 7)
    assert torch.all(row["action"] == 0.5)
    assert row["action_is_pad"].sum() == 49
    assert row["task"] == task["language"]
    obs = {
        "robot0_eef_pos": state[:3],
        "robot0_eef_quat": [0, 0, np.sqrt(0.5), np.sqrt(0.5)],
        "robot0_gripper_qpos": state[6:],
        "agentview_image": images[1, ::-1].copy(),
        "robot0_eye_in_hand_image": images[1, ::-1].copy(),
    }
    live = observation_batch(obs, task, "cpu")
    torch.testing.assert_close(row["state"], live["state"][0])
    torch.testing.assert_close(row["agentview_rgb"], live["agentview_rgb"][0])


def test_pi0_stats_exclude_validation(monkeypatch, tmp_path):
    h5py = pytest.importorskip("h5py")
    pytest.importorskip("torch")
    from roboscope.data import pi0

    path = tmp_path / "fixture.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": "move the bowl"})
        for name, value in (("demo_0", 1), ("demo_1", 1000)):
            group = data.create_group(name)
            group["actions"] = np.full((2, 7), value, dtype=np.float32)
            group["obs/ee_states"] = np.full((2, 6), value, dtype=np.float32)
            group["obs/gripper_states"] = np.full((2, 2), value, dtype=np.float32)
    manifest = {
        "tasks": [
            {
                "name": "fixture",
                "path": str(path),
                "episodes": [
                    {"demo": "demo_0", "length": 2, "split": "train"},
                    {"demo": "demo_1", "length": 2, "split": "val"},
                ],
            }
        ]
    }
    monkeypatch.setattr(pi0, "prepare", lambda cfg: manifest)
    result = pi0.prepare_pi0({})
    assert result["pi0_state_mean"] == [1] * 8
    assert result["pi0_action_mean"] == [1] * 7
