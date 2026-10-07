"""HF adapter semantics, checkpoint identity and 8D learner compatibility."""

import json
import os
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")

from roboscope.envs.hf_libero import compact_observation  # noqa: E402
from roboscope.policies.smolvla_hf import checkpoint_files, validate_config  # noqa: E402
from roboscope.rl.learner import RLTAgent  # noqa: E402
from roboscope.workflows.config import validate_rlt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def recipe():
    return json.loads((ROOT / "configs/libero_spatial/smolvla_rlt_hf.json").read_text())


def test_native_image_orientation_and_eef_state():
    image = np.arange(256 * 256 * 3, dtype=np.int64).reshape(256, 256, 3).astype(np.uint8)
    obs = {
        "pixels": {"image": image, "image2": image.copy()},
        "robot_state": {
            "eef": {"pos": np.array([1, 2, 3]), "quat": np.array([0, 0, np.sqrt(0.5), np.sqrt(0.5)])},
            "gripper": {"qpos": np.array([0.02, -0.02])},
            "joints": {"pos": np.full(7, 999)},
        },
    }
    result = compact_observation(obs)
    np.testing.assert_array_equal(result["agentview_rgb"], image[::-1, ::-1])
    np.testing.assert_array_equal(result["eye_in_hand_rgb"], image[::-1, ::-1])
    np.testing.assert_allclose(result["state"], [1, 2, 3, 0, 0, np.pi / 2, 0.02, -0.02], atol=1e-6)
    # Demonstration frames bypass this simulator adapter, so they are never flipped twice.
    assert result["state"].shape == (8,)


def test_hf_frames_map_language_to_native_task_id(monkeypatch):
    import lerobot.datasets.lerobot_dataset as module

    from roboscope.data.smolvla_hf import HFSpatialFrames

    pixels = torch.arange(3 * 256 * 256).remainder(256).to(torch.uint8).reshape(3, 256, 256)
    sample = {
        "task": "on ramekin",
        "task_index": 7,
        "observation.state": torch.arange(8).float(),
        "observation.images.image": pixels.float() / 255,
        "observation.images.image2": pixels.float() / 255,
    }
    monkeypatch.setattr(module, "LeRobotDataset", lambda *args, **kwargs: [sample])
    ds = HFSpatialFrames(
        {"dataset_repo": "local", "dataset_root": "/unused", "tasks": [{"id": 5, "language": "on ramekin"}]}
    )
    assert ds[0]["task_id"].item() == 5
    torch.testing.assert_close(ds[0]["agentview_rgb"], pixels, rtol=0, atol=0)
    torch.testing.assert_close(ds[0]["state"], sample["observation.state"])


def test_checkpoint_identity_includes_saved_normalization(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"weights")
    for kind in ("pre", "post"):
        (tmp_path / f"policy_{kind}processor.json").write_text(
            json.dumps({"steps": [{"state_file": f"{kind}.safetensors"}]})
        )
        (tmp_path / f"{kind}.safetensors").write_bytes(b"stats")
    original = checkpoint_files(tmp_path)
    (tmp_path / "post.safetensors").write_bytes(b"different action statistics")
    assert checkpoint_files(tmp_path) != original
    assert checkpoint_files(tmp_path)["model.safetensors"] == original["model.safetensors"]
    (tmp_path / "pre.safetensors").unlink()
    with pytest.raises(FileNotFoundError):
        checkpoint_files(tmp_path)


def test_joint_checkpoint_cannot_be_used_as_hf_sft():
    config = {"input_features": {"observation.state": {"shape": [9]}}}
    with pytest.raises(ValueError, match="8D EEF"):
        validate_config(config)


@pytest.mark.parametrize(
    "change",
    [{"state_dim": 9}, {"image_size": 128}, {"environment_backend": "legacy"}, {"environment_versions": {}}],
)
def test_hf_protocol_cannot_silently_fall_back(change):
    with pytest.raises(ValueError):
        validate_rlt({**recipe(), **change})


def test_hf_actor_and_critic_update_with_eight_dimensional_proprio():
    cfg = {**recipe(), "token_dim": 16, "hidden_dims": [16, 16]}
    validate_rlt(cfg)
    agent = RLTAgent(cfg)
    dim, horizon, batch = 16 + 8 + 1, cfg["action_horizon"], 2
    row = {
        "state": torch.randn(batch, dim),
        "next_state": torch.randn(batch, dim),
        "reference": torch.zeros(batch, horizon, 7),
        "next_reference": torch.zeros(batch, horizon, 7),
        "action": torch.zeros(batch, horizon, 7),
        "action_mask": torch.ones(batch, horizon),
        "reward": torch.ones(batch),
        "discount": torch.zeros(batch),
    }
    agent.update(row)
    metrics = agent.update(row)
    assert np.isfinite(metrics["actor_loss"]) and np.isfinite(metrics["critic_loss"])


@pytest.mark.local
@pytest.mark.skipif(not os.environ.get("ROBOSCOPE_HF_RESET_RUN"), reason="Requires HF assets and EGL binding")
@pytest.mark.parametrize("task_id", [0, 4])
def test_fixed_initial_state_independent_of_reset_history(tmp_path, task_id):
    from roboscope.envs.hf_libero import FIXED_RESET_PROTOCOL, reset_fixed_state, reset_state_digest, setup

    source = Path(os.environ["ROBOSCOPE_HF_RESET_RUN"])
    cfg = json.loads((source / "env_config.json").read_text())
    suite = setup(cfg, tmp_path)
    from lerobot.envs.libero import LiberoEnv

    def new_env():
        return LiberoEnv(
            suite,
            task_id,
            "libero_spatial",
            obs_type="pixels_agent_pos",
            observation_width=256,
            observation_height=256,
            init_states=True,
            episode_length=cfg["rollout_horizon"],
            num_steps_wait=cfg["settle_steps"],
            hard_reset=True,
        )

    env = new_env()
    reference = None
    try:
        for initial in (7, 0, 25, 7):
            env.init_state_id = initial
            obs, _ = reset_fixed_state(env, cfg["eval_seed"] + task_id * 1000 + initial, FIXED_RESET_PROTOCOL)
            assert np.isfinite(obs["robot_state"]["eef"]["pos"]).all()
            assert len(env._env.env.object_property_initializers) == (1 if task_id == 4 else 0)
            if initial == 7:
                state = reset_state_digest(env)
                if reference is None:
                    reference = state
                else:
                    assert state == reference
    finally:
        env.close()
    # A different worker/fresh instance must produce the identical settled scene.
    env = new_env()
    try:
        env.init_state_id = 7
        reset_fixed_state(env, cfg["eval_seed"] + task_id * 1000 + 7, FIXED_RESET_PROTOCOL)
        assert reset_state_digest(env) == reference
    finally:
        env.close()
