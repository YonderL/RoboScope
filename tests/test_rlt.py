import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from roboscope.rl.collector import collect_episode  # noqa: E402
from roboscope.rl.learner import RLTAgent, bellman_target  # noqa: E402
from roboscope.rl.replay import ReplayBuffer, episode_transitions  # noqa: E402
from roboscope.rl.token import RLToken  # noqa: E402
from roboscope.trainers import smolvla_rlt as trainer  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def config():
    cfg = json.loads((ROOT / "configs/libero_spatial/smolvla_rlt.json").read_text())
    cfg.update(
        token_dim=16,
        token_heads=4,
        token_layers=1,
        hidden_dims=[16, 16],
        batch_size=4,
        replay_capacity=100,
        initial_updates=2,
        updates_per_transition=1,
        warmup_steps=12,
        online_steps=40,
        rollout_horizon=12,
        amp=False,
    )
    return cfg


@pytest.mark.parametrize(("language_length", "state_length", "padding"), [(5, 1, 0), (8, 2, 4)])
def test_image_feature_selection_excludes_language_state_and_padding(language_length, state_length, padding):
    from roboscope.policies.smolvla_rlt import image_features_from_prefix

    image_length = 6  # two cameras, three tokens each
    state_start = image_length + language_length
    length = state_start + state_length + padding
    features = torch.arange(2 * length * 4).float().reshape(2, length, 4).requires_grad_()
    valid = torch.ones(2, length, dtype=torch.bool)
    valid[0, :3] = False  # missing camera must not shift the language/image boundary
    valid[:, image_length + 2 : state_start] = False  # language padding still occupies positions
    valid[:, state_start + state_length :] = False
    attention = torch.zeros(2, length, dtype=torch.bool)
    attention[:, state_start : state_start + state_length] = True
    selected, mask = image_features_from_prefix(features, valid, attention, language_length)
    torch.testing.assert_close(selected, features[:, :image_length])
    assert torch.equal(mask, valid[:, :image_length])
    assert not selected.requires_grad
    altered = features.detach().clone()
    altered[:, image_length:] = 1e6
    same, _ = image_features_from_prefix(altered, valid, attention, language_length)
    torch.testing.assert_close(same, selected, rtol=0, atol=0)


def test_rl_token_bottleneck_and_causal_decoder():
    torch.set_num_threads(2)
    # Narrow width keeps Linear projections for unit-test bottlenecks only.
    model = RLToken(12, 16, 1, 4)
    features = torch.randn(2, 7, 12, requires_grad=True)
    valid = torch.ones(2, 7, dtype=torch.bool)
    valid[:, 3] = False
    encoded = model.encode(features, valid)
    changed = features.detach().clone()
    changed[:, 3] = 999
    torch.testing.assert_close(model.encode(changed, valid), encoded)
    original = model.reconstruct(encoded.detach(), features, valid)
    # Changing target i or any future token cannot influence prediction i.
    changed = features.detach().clone()
    changed[:, 4:] = torch.randn_like(changed[:, 4:]) * 100
    altered = model.reconstruct(encoded.detach(), changed, valid)
    torch.testing.assert_close(original[:, :5], altered[:, :5])
    model(features, valid).backward()
    assert features.grad is None
    assert model.readout.grad.abs().sum() > 0
    assert model.input.weight.grad.abs().sum() > 0


def test_rl_token_matching_width_skips_input_projection():
    torch.set_num_threads(2)
    model = RLToken(16, 16, 1, 4)
    assert model.input is None and model.decoder_input is None
    assert isinstance(model.output, torch.nn.Linear)
    features = torch.randn(2, 5, 16, requires_grad=True)
    valid = torch.ones(2, 5, dtype=torch.bool)
    token = model.encode(features, valid)
    assert token.shape == (2, 16)
    original = model.reconstruct(token.detach(), features, valid)
    future = features.detach().clone()
    future[:, 3:] = torch.randn_like(future[:, 3:]) * 100
    altered = model.reconstruct(token.detach(), future, valid)
    torch.testing.assert_close(original[:, :4], altered[:, :4])
    model(features, valid).backward()
    assert features.grad is None
    assert model.readout.grad.abs().sum() > 0
    assert model.output.weight.grad.abs().sum() > 0


def rows(success=True, episode_id=0):
    cfg = config()
    features = {i: (np.full(26, i, np.float32), np.full((10, 7), i / 100, np.float32)) for i in range(12)}
    return list(episode_transitions(np.ones((12, 7)), success, features, cfg, 0, episode_id, 123, True))


def test_chunk_rewards_duration_terminals_and_replay_resume():
    cfg = config()
    success, failure = rows(), rows(False, 1)
    assert len(success) == 6
    assert [row["duration"] for row in success] == [10, 10, 8, 6, 4, 2]
    assert success[0]["discount"] == pytest.approx(cfg["gamma"] ** 10)
    assert success[0]["reward"] == 0
    assert success[1]["reward"] == pytest.approx(cfg["gamma"] ** 9)
    assert all(row["discount"] == 0 for row in success[1:])
    assert all(row["discount"] == 0 and row["truncated"] for row in failure[1:])
    assert success[-1]["action_mask"].sum() == 2
    assert np.all(success[-1]["action"][2:] == 0)
    assert np.all(success[0]["next_reference"] == 0.10)
    replay = ReplayBuffer(9, seed=4)
    for row in success + failure:
        replay.add(row)
    assert len(replay) == 9
    assert set(replay.arrays["episode_id"][:9]) == {0, 1}
    saved = replay.state_dict()
    restored = ReplayBuffer(9)
    restored.load_state_dict(saved)
    for key, value in replay.sample(20, "cpu").items():
        # Rewind once so all fields compare against the same sampled indices.
        restored.load_state_dict(saved)
        torch.testing.assert_close(value, restored.sample(20, "cpu")[key], rtol=0, atol=0)


def test_twin_q_delayed_actor_reference_dropout_and_targets():
    torch.set_num_threads(2)
    cfg = config()
    agent = RLTAgent(cfg)
    replay = ReplayBuffer(100)
    for row in rows() + rows(False, 1):
        replay.add(row)
    batch = replay.sample(4, "cpu")
    original = copy.deepcopy(agent.actor.state_dict())
    target_before = copy.deepcopy(agent.target_critic.state_dict())
    metrics = agent.update(batch)
    assert "actor_loss" not in metrics
    for key, value in agent.actor.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    for key, value in agent.target_critic.state_dict().items():
        torch.testing.assert_close(
            value, target_before[key].lerp(agent.critic.state_dict()[key], cfg["target_tau"])
        )
    metrics = agent.update(batch)
    assert "actor_loss" in metrics and np.isfinite(list(metrics.values())).all()
    assert any(not torch.equal(value, original[key]) for key, value in agent.actor.state_dict().items())
    assert all(p.grad is None for p in agent.target_critic.parameters())
    torch.testing.assert_close(
        agent.actor(batch["state"], batch["reference"], dropout=1),
        agent.actor(batch["state"], torch.zeros_like(batch["reference"])),
    )
    target = bellman_target(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.9]), torch.tensor([10.0, 2.0]))
    torch.testing.assert_close(target, torch.tensor([1.0, 1.8]))


class FakePool:
    def __init__(self):
        self.connections = [self]
        self.commands = []

    def send(self, message):
        self.commands.append(message)
        if message[0] == "reset_training":
            self.step = 0
        else:
            self.step += 1

    def receive(self, slot):
        return {
            "state": np.full(9, self.step, np.float32),
            "agentview_rgb": np.zeros((4, 4, 3), np.uint8),
            "eye_in_hand_rgb": np.zeros((4, 4, 3), np.uint8),
        }, self.step == 12


def test_rollout_references_are_aligned_and_only_executed_actions_stored():
    cfg = config()

    class Policy:
        def describe(self, batch, noise):
            t = batch["state"][:, 0]
            return t[:, None].expand(-1, 26), t[:, None, None].expand(-1, 10, 7) / 100

    pool = FakePool()
    transitions, metrics = collect_episode(pool, Policy(), {"id": 0}, 0, cfg, torch.device("cpu"), True)
    assert pool.commands[0][0] == "reset_training"
    assert metrics["steps"] == 12 and metrics["success"] == 1
    assert transitions[1]["reference"][0, 0] == pytest.approx(0.02)
    assert transitions[0]["next_reference"][0, 0] == pytest.approx(0.10)
    assert np.all(transitions[0]["action"] == 0)
    assert transitions[1]["action"][8, 0] == pytest.approx(0.10)
    assert all(row["episode_id"] == 0 for row in transitions)


@pytest.mark.parametrize("separate_warmup", [False, True])
def test_online_episode_boundary_resume_restores_agent_buffer_and_rng(tmp_path, monkeypatch, separate_warmup):
    cfg = config()
    manifest = {"tasks": [{"id": 0}, {"id": 1}]}

    def policy():
        return SimpleNamespace(base=torch.nn.Linear(1, 1), token=torch.nn.Linear(1, 1), eval=lambda: None)

    # Synthetic collector exercises real actor/critic/optimizers and checkpointing.
    def collect(pool, policy, task, episode_id, cfg, device, warmup):
        data = rows(episode_id % 2 == 0, episode_id)
        return data, {
            "episode_id": episode_id,
            "task_id": task["id"],
            "steps": 12,
            "success": int(episode_id % 2 == 0),
            "warmup": warmup,
        }

    initial = policy()
    full, restart = tmp_path / "full", tmp_path / "restart"
    full.mkdir()
    restart.mkdir()
    torch.manual_seed(14)
    trainer.train_online(
        full, cfg, manifest, copy.deepcopy(initial), None, torch.device("cpu"), collector=collect
    )
    atomic = trainer.atomic_save

    def interrupt(path, payload):
        atomic(path, payload)
        if Path(path).name == "last.pt" and payload["episode_id"] == 2:
            raise InterruptedError

    torch.manual_seed(14)
    if separate_warmup:
        trainer.train_online(
            restart,
            cfg,
            manifest,
            copy.deepcopy(initial),
            None,
            torch.device("cpu"),
            collector=collect,
            collect_only=True,
        )
        warmup = torch.load(restart / "last.pt", weights_only=False)
        assert warmup["env_steps"] == cfg["warmup_steps"]
        assert warmup["agent"]["updates"] == 0 and not warmup["warmup_done"]
        assert warmup["replay"]["size"] > 0 and not (restart / "final.pt").exists()
        # A resumed warmup must neither recollect episodes nor train the actor.
        again = trainer.train_online(
            restart,
            cfg,
            manifest,
            copy.deepcopy(initial),
            None,
            torch.device("cpu"),
            collector=collect,
            collect_only=True,
            resume=True,
        )
        assert again == warmup["records"]
    else:
        monkeypatch.setattr(trainer, "atomic_save", interrupt)
        with pytest.raises(InterruptedError):
            trainer.train_online(
                restart, cfg, manifest, copy.deepcopy(initial), None, torch.device("cpu"), collector=collect
            )
    monkeypatch.setattr(trainer, "atomic_save", atomic)
    trainer.train_online(
        restart,
        cfg,
        manifest,
        copy.deepcopy(initial),
        None,
        torch.device("cpu"),
        resume=True,
        collector=collect,
    )
    expected = torch.load(full / "last.pt", weights_only=False)
    actual = torch.load(restart / "last.pt", weights_only=False)
    assert actual["records"] == expected["records"]
    for key, value in expected["agent"]["model"].items():
        torch.testing.assert_close(value, actual["agent"]["model"][key], rtol=0, atol=0)
    for key, value in expected["replay"]["arrays"].items():
        np.testing.assert_array_equal(value, actual["replay"]["arrays"][key])
    with pytest.raises(ValueError, match="cannot return to warmup"):
        trainer.train_online(
            restart,
            cfg,
            manifest,
            copy.deepcopy(initial),
            None,
            torch.device("cpu"),
            collector=collect,
            collect_only=True,
            resume=True,
        )


def test_token_stage_resumes_and_never_updates_sft(tmp_path, monkeypatch):
    cfg = {
        **config(),
        "token_steps": 4,
        "token_save_every": 1,
        "token_batch_size": 4,
        "feature_batch_size": 2,
        "workers": 0,
        "log_every": 1,
    }

    class Dataset(torch.utils.data.Dataset):
        def __len__(self):
            return 13

        def __getitem__(self, index):
            return {"state": torch.full((12,), index / 13)}

    def features(base, batch):
        with torch.no_grad():
            return base(batch["state"])[:, None].expand(-1, 3, -1), torch.ones(
                len(batch["state"]), 3, dtype=torch.bool
            )

    monkeypatch.setattr(trainer, "FrameDataset", lambda *args: Dataset())
    monkeypatch.setattr(trainer, "vlm_features", features)
    initial = SimpleNamespace(base=torch.nn.Linear(12, 12), token=RLToken(12, 16, 1, 4))
    original = copy.deepcopy(initial.base.state_dict())
    full, resumed = tmp_path / "full_token", tmp_path / "resumed_token"
    full.mkdir()
    resumed.mkdir()
    model = copy.deepcopy(initial)
    trainer.train_token(full, cfg, {}, model, torch.device("cpu"))
    for key, value in original.items():
        torch.testing.assert_close(model.base.state_dict()[key], value, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in model.base.parameters())
    atomic = trainer.atomic_save

    def interrupt(path, payload):
        atomic(path, payload)
        if Path(path).name == "token_last.pt" and payload["step"] == 2:
            raise InterruptedError

    monkeypatch.setattr(trainer, "atomic_save", interrupt)
    with pytest.raises(InterruptedError):
        trainer.train_token(resumed, cfg, {}, copy.deepcopy(initial), torch.device("cpu"))
    monkeypatch.setattr(trainer, "atomic_save", atomic)
    trainer.train_token(resumed, cfg, {}, copy.deepcopy(initial), torch.device("cpu"), resume=True)
    expected = torch.load(full / "token.pt", weights_only=False)
    actual = torch.load(resumed / "token.pt", weights_only=False)
    for key, value in expected["token"].items():
        torch.testing.assert_close(value, actual["token"][key], rtol=0, atol=0)


@pytest.mark.parametrize("failure", ["loss", "gradient"])
def test_nonfinite_token_update_cannot_be_exported_as_completed(tmp_path, monkeypatch, failure):
    cfg = {
        **config(),
        "token_steps": 1,
        "token_save_every": 1,
        "token_batch_size": 2,
        "feature_batch_size": 2,
        "workers": 0,
        "log_every": 1,
    }
    monkeypatch.setattr(
        trainer, "FrameDataset", lambda *args: [{"state": torch.ones(4)}, {"state": torch.ones(4)}]
    )
    monkeypatch.setattr(
        trainer,
        "vlm_features",
        lambda base, batch: (batch["state"][:, None], torch.ones(2, 1, dtype=torch.bool)),
    )

    class Token(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(1.0))
            if failure == "gradient":
                self.weight.register_hook(lambda grad: grad * float("nan"))

        def forward(self, features, valid):
            return self.weight * (float("nan") if failure == "loss" else 1.0)

    token = Token()
    model = SimpleNamespace(base=torch.nn.Linear(4, 4), token=token)
    with pytest.raises((FloatingPointError, RuntimeError), match="[Nn]on.?finite"):
        trainer.train_token(tmp_path, cfg, {}, model, torch.device("cpu"))
    assert token.weight.item() == 1.0
    assert not (tmp_path / "token.pt").exists()
    assert not (tmp_path / "token_last.pt").exists()


def check_native_smolvla_rlt(base, batch, manifest, device, tmp_path):
    """Called by the native tiny SmolVLA test on both CPU and optional CUDA."""
    from roboscope.policies.smolvla_rlt import build_policy, load_policy, vlm_features
    from roboscope.runtime.common import digest
    from roboscope.trainers.pi0 import manifest_digest

    cfg = config()
    cfg["rollout_horizon"] = 600
    # Tiny native SmolVLM uses hidden_size=32; production requires matching width.
    cfg["token_dim"] = base.policy.model.vlm_with_expert.config.text_config.hidden_size
    cfg["token_heads"] = 4
    base.zero_grad(set_to_none=True)
    policy = build_policy(base, cfg, manifest).to(device)
    features, valid = vlm_features(base, batch)
    assert features.ndim == 3 and features.shape[:2] == valid.shape
    assert not features.requires_grad
    # Check image-only extraction against independent native visual token counts
    # and native full-prefix outputs, while retaining all language/state in the VLA.
    from lerobot.policies.common.vla_utils import make_att_2d_masks

    with torch.no_grad():
        prepared = base.prepare_batch(batch)
        images, image_masks = base.policy.prepare_images(prepared)
        model = base.policy.model
        count = sum(model.vlm_with_expert.embed_image(img).shape[1] for img in images)
        embedded, full_valid, attention = model.embed_prefix(
            images,
            image_masks,
            prepared["observation.language.tokens"],
            prepared["observation.language.attention_mask"],
            state=base.policy.prepare_state(prepared),
        )
        outputs, _ = model.vlm_with_expert.forward(
            attention_mask=make_att_2d_masks(full_valid, attention),
            position_ids=full_valid.cumsum(-1) - 1,
            inputs_embeds=[embedded, None],
            use_cache=True,
        )
    assert features.shape[1] == count < full_valid.shape[1]
    torch.testing.assert_close(features, outputs[0][:, :count].float(), rtol=0, atol=0)
    assert torch.equal(valid, full_valid[:, :count])
    changed_state = {**batch, "state": batch["state"] + 1}
    image_only, _ = vlm_features(base, changed_state)
    torch.testing.assert_close(image_only, features, rtol=0, atol=0)
    loss = policy.token(features, valid)
    loss.backward()
    assert policy.token.readout.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in base.parameters())
    policy.eval()
    noise = torch.randn(2, 50, 32, device=device)
    batch = {**batch, "remaining_steps": torch.tensor([600, 100], device=device)}
    state, reference = policy.describe(batch, noise)
    expected = base.predict(batch, noise)[:, :10].clamp(-1, 1).to(device)
    torch.testing.assert_close(reference, expected, rtol=0, atol=0)
    assert state.shape == (2, cfg["token_dim"] + 10)
    torch.testing.assert_close(
        state[:, cfg["token_dim"] : -1],
        (batch["state"].to(device) - policy.state_mean) / policy.state_std,
    )
    torch.testing.assert_close(state[:, -1], torch.tensor([1.0, 1 / 6], device=device))
    assert policy.predict(batch, noise).shape == (2, 10, 7)
    with pytest.raises(ValueError, match="retrain old"):
        build_policy(base, {k: v for k, v in cfg.items() if k != "token_features"}, manifest)
    with pytest.raises(ValueError, match="must equal VLM hidden_size"):
        build_policy(base, {**cfg, "token_dim": max(8, cfg["token_dim"] // 2)}, manifest)
    with torch.autocast(device.type, dtype=torch.bfloat16):
        assert torch.isfinite(policy.predict(batch, noise)).all()
    saved = tmp_path / "rlt_tiny.pt"
    source = tmp_path / "sft"
    source.mkdir()
    (source / "manifest.json").write_text(json.dumps(manifest))
    sft = source / "final.pt"
    torch.save(
        {
            "format": "roboscope.smolvla.v1",
            "config": {"pretrained_path": str(tmp_path), "vlm_path": str(tmp_path)},
            "manifest_sha256": manifest_digest(manifest),
            "model": base.state_dict(),
        },
        sft,
    )
    cfg.update(sft_source=str(source), sft_checkpoint=str(sft), sft_sha256=digest(sft))
    torch.save(
        {
            "format": "roboscope.smolvla_rlt.v1",
            "config": cfg,
            "token": policy.token.state_dict(),
            "actor": policy.actor.state_dict(),
            "env_steps": 12,
        },
        saved,
    )
    reloaded, steps = load_policy(saved, manifest, {}, device)
    assert steps == 12 and all(not p.requires_grad for p in reloaded.parameters())
    torch.testing.assert_close(reloaded.predict(batch, noise), policy.predict(batch, noise), rtol=0, atol=0)
    evaluation_manifest = copy.deepcopy(manifest)
    evaluation_manifest["tasks"][0]["eval_initial_state_ids"] = [0]
    load_policy(saved, evaluation_manifest, {}, device)
    changed = {**manifest, "state_mean": [2.0] * 9}
    with pytest.raises(ValueError, match="differ beyond"):
        load_policy(saved, changed, {}, device)
    (source / "manifest.json").write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="manifest changed"):
        load_policy(saved, manifest, {}, device)
    (source / "manifest.json").write_text(json.dumps(manifest))
    # Opt-in integration uses real LIBERO assets but tiny random model weights;
    # it verifies simulator/replay/update plumbing, never a benchmark score.
    if os.environ.get("ROBOSCOPE_RLT_LIBERO_RUN"):
        from roboscope.envs.pool import EnvPool

        source = Path(os.environ["ROBOSCOPE_RLT_LIBERO_RUN"])
        environment = json.loads((source / "env_config.json").read_text())
        task = json.loads((source / "manifest.json").read_text())["tasks"][0]
        smoke = {
            **environment,
            **cfg,
            "rollout_horizon": 12,
            "feature_batch_size": 2,
            "amp": True,
            "warmup_steps": 12,
            "online_steps": 24,
        }
        policy.cfg = smoke
        pool = EnvPool(smoke, tmp_path, 1)
        try:
            run = tmp_path / "real_libero_stages"
            run.mkdir()
            trainer.train_online(run, smoke, {"tasks": [task]}, policy, pool, device, collect_only=True)
            warmup = torch.load(run / "last.pt", weights_only=False)
            assert warmup["updates"] == 0 and warmup["replay"]["size"] == 6
            trainer.train_online(run, smoke, {"tasks": [task]}, policy, pool, device, resume=True)
            final = torch.load(run / "last.pt", weights_only=False)
            assert final["env_steps"] == 24 and final["replay"]["size"] == 12
            assert final["updates"] == 14 and (run / "final.pt").exists()
            assert [row["warmup"] for row in final["records"]] == [True, False]
            assert all(torch.isfinite(value).all() for value in final["agent"]["model"].values())
            print({"real_libero_staged_updates": final["updates"]}, flush=True)
        finally:
            pool.close()
