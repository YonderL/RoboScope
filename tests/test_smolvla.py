import copy
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: E402

from roboscope.trainers import smolvla  # noqa: E402


class TinyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 13

    def __getitem__(self, index):
        return {
            "state": torch.tensor([index / 13, 0.5]),
            "action": torch.ones(4, 1),
            "action_is_pad": torch.arange(4) > index % 4,
        }


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.2))
        self.policy = SimpleNamespace(config=SmolVLAConfig(device="cpu"))

    def forward(self, batch):
        mask = ~batch["action_is_pad"]
        prediction = batch["state"][:, :1] * self.weight + torch.randn_like(batch["action"][..., 0]) * 0.01
        return ((prediction - batch["action"][..., 0]).square() * mask).sum() / mask.sum()


def test_accumulation_weights_valid_actions_not_microbatch_count():
    class Deterministic(TinyPolicy):
        def forward(self, batch):
            mask = ~batch["action_is_pad"]
            return ((batch["state"][:, :1] * self.weight).square() * mask).sum() / mask.sum()

    batch = torch.utils.data.default_collate([TinyDataset()[i] for i in range(4)])
    full, micro = Deterministic(), Deterministic()
    full(batch).backward()
    smolvla.backward_batch(micro, batch, 2, torch.device("cpu"))
    torch.testing.assert_close(full.weight.grad, micro.weight.grad)


def test_native_optimizer_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    cfg = dict(
        seed=0,
        train_steps=4,
        batch_size=4,
        micro_batch_size=2,
        workers=0,
        validation_samples=5,
        validate_every=2,
        save_every=1,
        log_every=1,
    )
    dataset, device = TinyDataset(), torch.device("cpu")
    smolvla.train(tmp_path / "full", cfg, {}, TinyPolicy(), dataset, dataset, device)
    atomic = smolvla.atomic_save

    def interrupt(path, payload):
        atomic(path, payload)
        if Path(path).name == "last.pt" and payload["step"] == 2:
            raise InterruptedError

    monkeypatch.setattr(smolvla, "atomic_save", interrupt)
    with pytest.raises(InterruptedError):
        smolvla.train(tmp_path / "restart", cfg, {}, TinyPolicy(), dataset, dataset, device)
    monkeypatch.setattr(smolvla, "atomic_save", atomic)
    smolvla.train(tmp_path / "restart", cfg, {}, TinyPolicy(), dataset, dataset, device, resume=True)
    full = torch.load(tmp_path / "full/last.pt", weights_only=False)
    resumed = torch.load(tmp_path / "restart/last.pt", weights_only=False)
    torch.testing.assert_close(full["model"]["weight"], resumed["model"]["weight"], rtol=0, atol=0)
    assert full["history"] == resumed["history"]
    assert full["scheduler"] == resumed["scheduler"]
    with pytest.raises(ValueError, match="changed"):
        smolvla.train(
            tmp_path / "restart", cfg, {"changed": True}, TinyPolicy(), dataset, dataset, device, resume=True
        )


@pytest.mark.parametrize(("kind", "horizon"), [("smolvla", 50), ("rlt", 10)])
def test_shared_rollout_native_chunk_and_episode_rng(tmp_path, monkeypatch, kind, horizon):
    from roboscope.evaluation import worker

    class Connection:
        def send(self, message):
            command, payload = message
            if command == "reset":
                self.task, self.initial, _ = payload
                self.steps = 0
            else:
                self.steps += 1

    class Pool:
        def __init__(self, cfg, directory, size):
            self.connections = [Connection() for _ in range(size)]

        def receive(self, slot):
            item = self.connections[slot]
            obs = {"state": np.zeros(9), "eef_pos": np.zeros(3)}
            return obs, item.steps >= (17 if item.initial == 0 else 101)

        def close(self):
            pass

    class Model:
        def predict(self, batch, noise):
            assert noise.shape[1:] == (50, 32)
            return noise[..., :7]

    monkeypatch.setattr(worker, "EnvPool", Pool)
    monkeypatch.setattr(worker, "batch_observations", lambda *args: {})
    monkeypatch.setattr(
        worker,
        "torch",
        SimpleNamespace(
            bfloat16=torch.bfloat16,
            tensor=lambda data, **kwargs: torch.tensor(data),
            Generator=lambda **kwargs: torch.Generator(),
            stack=torch.stack,
            randn=lambda *args, **kwargs: torch.randn(*args, generator=kwargs["generator"]),
            inference_mode=torch.inference_mode,
            autocast=lambda *args, **kwargs: nullcontext(),
        ),
    )
    cfg = dict(eval_seed=10000, eval_envs=2, video_episodes_per_task=0, rollout_horizon=600, amp=True)
    jobs = [({"id": 0, "name": "fixture"}, i) for i in range(3)]
    traces = []
    for size in (1, 2):
        directory = tmp_path / str(size)
        directory.mkdir()
        rows = []
        worker.rollout(
            Model(),
            {**cfg, "eval_envs": size},
            {},
            jobs,
            {"model": kind, "ta": horizon},
            directory,
            rows.append,
        )
        rows.sort(key=lambda row: row["initial_state_id"])
        assert [row["model_calls"] for row in rows] == [
            (length + horizon - 1) // horizon for length in (17, 101, 101)
        ]
        assert [row["steps"] for row in rows] == [17, 101, 101]
        traces.append([np.load(row["trace"])["raw_actions"] for row in rows])
        assert np.load(rows[1]["trace"])["query_steps"].tolist() == list(range(0, 101, horizon))
    for first, second in zip(*traces, strict=True):
        np.testing.assert_array_equal(first, second)


def test_dataset_cameras_language_and_episode_padding(tmp_path):
    h5py = pytest.importorskip("h5py")
    from roboscope.data.smolvla import SmolVLADataset
    from roboscope.envs.pool import compact_observation

    path = tmp_path / "fixture.hdf5"
    images = np.zeros((2, 128, 128, 3), dtype=np.uint8)
    images[:, 0, :, :] = 255
    with h5py.File(path, "w") as handle:
        for index in range(2):
            demo = handle.create_group(f"data/demo_{index}")
            demo["actions"] = np.full((2, 7), index + 0.5, dtype=np.float32)
            demo["obs/joint_states"] = np.ones((2, 7), dtype=np.float32)
            demo["obs/gripper_states"] = np.zeros((2, 2), dtype=np.float32)
            for key in ("agentview_rgb", "eye_in_hand_rgb"):
                demo[f"obs/{key}"] = images
    task = {
        "id": 0,
        "path": str(path),
        "language": "pick up bowl",
        "image_convention": "opengl",
        "episodes": [{"demo": f"demo_{i}", "length": 2, "split": "train"} for i in range(2)],
    }
    ds = SmolVLADataset({"tasks": [task]}, 50, "train")
    row = ds[1]
    assert row["task"] == "pick up bowl"
    assert row["action_is_pad"].sum() == 49 and torch.all(row["action"] == 0.5)
    assert torch.all(ds[2]["action"] == 1.5)
    live = compact_observation(
        {
            "robot0_eef_pos": np.zeros(3),
            "robot0_joint_pos": np.ones(7),
            "robot0_gripper_qpos": np.zeros(2),
            "agentview_image": images[1, ::-1],
            "robot0_eye_in_hand_image": images[1, ::-1],
        }
    )
    np.testing.assert_array_equal(row["state"].numpy(), live["state"])
    np.testing.assert_array_equal(row["agentview_rgb"].permute(1, 2, 0).numpy(), live["agentview_rgb"])


def test_native_smolvla_forward_processors_and_checkpoint_roundtrip(tmp_path, monkeypatch):
    """Real tiny SmolVLM/expert + processors; no remote assets or mocked policy math."""
    pytest.importorskip("transformers")
    from lerobot.policies.smolvla import smolvlm_with_expert
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy as NativePolicy
    from safetensors.torch import save_model
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import AutoTokenizer, PreTrainedTokenizerFast, SmolVLMConfig

    from roboscope.policies import smolvla as adapter

    torch.set_num_threads(2)
    device = torch.device(os.environ.get("ROBOSCOPE_SMOLVLA_TEST_DEVICE", "cpu"))
    tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "pick": 2, "bowl": 3}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]", unk_token="[UNK]")
    tokenizer.fake_image_token_id, tokenizer.global_image_token_id = 4, 5
    backbone = SmolVLMConfig(
        image_token_id=6,
        pad_token_id=0,
        scale_factor=2,
        vision_config=dict(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=16,
            patch_size=4,
        ),
        text_config=dict(
            model_type="llama",
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            pad_token_id=0,
        ),
    )
    monkeypatch.setattr(
        smolvlm_with_expert.AutoConfig, "from_pretrained", lambda *a, **kw: copy.deepcopy(backbone)
    )
    monkeypatch.setattr(
        smolvlm_with_expert.AutoProcessor,
        "from_pretrained",
        lambda *a, **kw: SimpleNamespace(tokenizer=tokenizer),
    )
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **kw: tokenizer)
    native = SmolVLAConfig(
        device="cpu",
        load_vlm_weights=True,
        num_vlm_layers=2,
        resize_imgs_with_padding=(16, 16),
        num_steps=2,
        pad_language_to="max_length",
        prefix_length=0,
    )
    monkeypatch.setattr(SmolVLAConfig, "from_pretrained", lambda *a, **kw: copy.deepcopy(native))
    cfg = dict(pretrained_path=str(tmp_path), vlm_path=str(tmp_path))
    manifest = {
        "tasks": [{"id": 0, "language": "pick bowl"}],
        "state_mean": [1.0] * 9,
        "state_std": [2.0] * 9,
        "action_mean": [0.25] * 7,
        "action_std": [0.5] * 7,
    }
    # First construct a tiny native checkpoint, then exercise strict loading.
    original = adapter.load_backend(cfg, initialize_pretrained=False)
    save_model(original, str(tmp_path / "model.safetensors"))
    model = adapter.SmolVLAPolicy(cfg, manifest).to(device)
    assert isinstance(model.policy, NativePolicy)
    batch = {
        "state": torch.ones(2, 9),
        "task_id": torch.zeros(2, dtype=torch.long),
        "agentview_rgb": torch.zeros(2, 3, 16, 16, dtype=torch.uint8),
        "eye_in_hand_rgb": torch.full((2, 3, 16, 16), 255, dtype=torch.uint8),
        "action": torch.full((2, 50, 7), 0.25),
        "action_is_pad": torch.zeros(2, 50, dtype=torch.bool),
    }
    batch["action_is_pad"][1, 1:] = True
    prepared = model.prepare_batch(batch, include_action=True)
    assert prepared["observation.state"].abs().sum() == 0
    assert prepared["action"].abs().sum() == 0
    assert prepared[adapter.IMAGE_KEYS[1]].min() == 1
    assert prepared["observation.language.tokens"].shape == (2, 48)
    assert torch.equal(prepared["action_is_pad"].cpu(), batch["action_is_pad"])
    model.train()
    loss = model(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.policy.model.action_out_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.policy.model.vlm_with_expert.vlm.parameters())
    noise = torch.randn(2, 50, 32, device=device)
    model.eval()
    prediction = model.predict(batch, noise=noise)
    expected = model.policy.predict_action_chunk(model.prepare_batch(batch), noise=noise) * 0.5 + 0.25
    torch.testing.assert_close(prediction, expected.cpu())
    assert prediction.shape == (2, 50, 7) and torch.isfinite(prediction).all()
    restored = adapter.SmolVLAPolicy(cfg, manifest, initialize_pretrained=False).to(device)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored.predict(batch, noise=noise), prediction, rtol=0, atol=0)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        assert torch.isfinite(restored.predict(batch, noise=noise)).all()
    from test_rlt import check_native_smolvla_rlt

    check_native_smolvla_rlt(restored, batch, manifest, device, tmp_path)
    (tmp_path / "model.safetensors").unlink()
    with pytest.raises(Exception, match="No such file|not found"):
        adapter.SmolVLAPolicy(cfg, manifest)
