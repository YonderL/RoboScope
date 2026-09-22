"""Small CPU checks of LoRA gradients, adapter recovery and Pi-0 IO contracts."""

import types

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from roboscope.policies.pi0 import PROJECTIONS, LoRALinear, Pi0Policy, inject_lora  # noqa: E402


class FakeTokenizer:
    def __call__(self, prompt, **kwargs):
        assert prompt.endswith("\n")
        return {
            "input_ids": torch.arange(4).unsqueeze(0),
            "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        }


def fake_model():
    root = nn.Module()
    root.model = nn.Module()
    root.model.paligemma_with_expert = nn.Module()
    pair = root.model.paligemma_with_expert
    for branch in ("paligemma.model.language_model", "gemma_expert.model"):
        target = pair
        for name in branch.split("."):
            if not hasattr(target, name):
                setattr(target, name, nn.Module())
            target = getattr(target, name)
        layer = nn.Module()
        layer.self_attn = nn.ModuleDict({name + "_proj": nn.Linear(4, 4) for name in "qkvo"})
        target.layers = nn.ModuleList([layer])
    pair.paligemma.model.vision_tower = nn.Linear(4, 4)
    for name in PROJECTIONS:
        setattr(root.model, name, nn.Linear(4, 4))
    return root


def bare_policy():
    policy = Pi0Policy.__new__(Pi0Policy)
    nn.Module.__init__(policy)
    policy.cfg = {}
    policy.tokenizer = FakeTokenizer()
    policy._tokens = {}
    for name, dim, fill in (
        ("state_mean", 8, 2),
        ("state_std", 8, 2),
        ("action_mean", 7, 3),
        ("action_std", 7, 2),
    ):
        policy.register_buffer(name, torch.full((dim,), fill, dtype=torch.float32))
    return policy


def batch():
    return {
        "state": torch.full((2, 8), 4.0),
        "action": torch.full((2, 50, 7), 5.0),
        "action_is_pad": torch.zeros(2, 50, dtype=torch.bool),
        "agentview_rgb": torch.full((2, 3, 16, 16), 255, dtype=torch.uint8),
        "eye_in_hand_rgb": torch.zeros(2, 3, 16, 16, dtype=torch.uint8),
        "task": ["pick up the black bowl", "put the bowl on the plate"],
    }


def test_lora_starts_as_base_and_backpropagates_with_checkpoint():
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(4)
    base = nn.Linear(5, 3)
    inputs = torch.randn(2, 5)
    expected = base(inputs).detach()
    adapter = LoRALinear(base, 2, 4)
    output = checkpoint(adapter, inputs, use_reentrant=False)
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    output.square().mean().backward()
    assert adapter.base.weight.grad is None
    assert adapter.lora_B.grad.abs().sum() > 0
    optimizer = torch.optim.AdamW([p for p in adapter.parameters() if p.requires_grad], lr=0.01)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    adapter(inputs).square().mean().backward()
    assert adapter.lora_A.grad.abs().sum() > 0


def test_injection_preserves_frozen_vision_and_adapter_roundtrip():
    policy = bare_policy()
    policy.policy = fake_model()
    policy.lora_targets = inject_lora(policy.policy, {"lora_rank": 2, "lora_alpha": 2})
    assert len(policy.lora_targets) == 8
    assert not any(
        p.requires_grad
        for p in policy.policy.model.paligemma_with_expert.paligemma.model.vision_tower.parameters()
    )
    for name in PROJECTIONS:
        assert all(p.requires_grad for p in getattr(policy.policy.model, name).parameters())
    saved = policy.trainable_state_dict()
    assert saved and not any(".base." in key or "vision_tower" in key for key in saved)
    with torch.no_grad():
        for param in policy.parameters():
            if param.requires_grad:
                param.add_(1)
    policy.load_trainable_state_dict(saved)
    for key, value in policy.trainable_state_dict().items():
        torch.testing.assert_close(value, saved[key])
    with pytest.raises(ValueError, match="Adapter keys differ"):
        policy.load_trainable_state_dict({})
    with pytest.raises(ValueError, match="preserve RNG"):
        inject_lora(fake_model(), {"lora_dropout": 0.1, "gradient_checkpointing": True})


def test_language_images_and_normalization_contract():
    policy = bare_policy()
    prepared = policy.prepare_batch(batch(), include_action=True)
    assert prepared["observation.state"].shape == (2, 8)
    assert prepared["observation.state"].unique().item() == 1
    assert prepared["action"].unique().item() == 1
    assert prepared["observation.images.base_0_rgb"].unique().item() == 1
    assert prepared["observation.images.left_wrist_0_rgb"].unique().item() == 0
    assert prepared["observation.language.attention_mask"].dtype == torch.bool
    assert len(policy._tokens) == 2
    invalid = batch()
    invalid["state"] = torch.zeros(2, 9)
    with pytest.raises(ValueError, match="end-effector"):
        policy.prepare_batch(invalid)


def test_forward_ignores_padded_action_dimensions_and_predict_unnormalizes():
    policy = bare_policy()

    class FakeCore:
        def sample_noise(self, shape, device):
            return torch.zeros(shape, device=device)

        def sample_time(self, size, device):
            return torch.zeros(size, device=device)

        def __call__(self, images, masks, tokens, token_masks, state, actions, noise, time):
            assert state.shape[-1] == actions.shape[-1] == 32
            assert torch.equal(state[:, 8:], torch.zeros_like(state[:, 8:]))
            assert torch.equal(actions[:, :, 7:], torch.zeros_like(actions[:, :, 7:]))
            losses = torch.full((2, 50, 32), 1000.0)
            losses[..., :7] = 2.0
            return losses

    policy.policy = types.SimpleNamespace(
        model=FakeCore(),
        _preprocess_images=lambda prepared: ([], []),
        prepare_state=lambda prepared: torch.nn.functional.pad(prepared["observation.state"], (0, 24)),
        prepare_action=lambda prepared: torch.nn.functional.pad(prepared["action"], (0, 25)),
        predict_action_chunk=lambda prepared, noise: noise[..., :7],
    )
    assert policy(batch()).item() == 2.0
    noise = torch.ones(2, 50, 32)
    assert policy.predict(batch(), noise=noise).unique().item() == 5.0
    with pytest.raises(ValueError, match="full padded shape"):
        policy.predict(batch(), noise=torch.zeros(2, 50, 7))


def test_public_tokenizer_uses_official_pi0_sentencepiece_format(tmp_path):
    spm = pytest.importorskip("sentencepiece")
    from roboscope.policies.pi0 import OpenPiTokenizer

    corpus = tmp_path / "corpus.txt"
    corpus.write_text("pick up the black bowl\nput the bowl on the plate\n" * 20)
    spm.SentencePieceTrainer.train(
        input=str(corpus),
        model_prefix=str(tmp_path / "tiny"),
        vocab_size=30,
        pad_id=0,
        eos_id=1,
        bos_id=2,
        unk_id=3,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    tokenizer = OpenPiTokenizer(tmp_path / "tiny.model")
    output = tokenizer("  pick_up the black bowl\n", max_length=48)
    expected = tokenizer.processor.encode("pick up the black bowl", add_bos=True)
    expected += tokenizer.processor.encode("\n")
    count = int(output["attention_mask"].sum())
    assert output["input_ids"][0, :count].tolist() == expected
    assert output["input_ids"].shape == (1, 48)
    assert not output["input_ids"][0, count:].any()


def test_cached_base_requires_pinned_revision_and_complete_files(tmp_path):
    from roboscope.policies.pi0 import cached_pretrained_directory

    revision = "a" * 40
    folder = tmp_path / "models--lerobot--pi0_base" / "snapshots" / revision
    folder.mkdir(parents=True)
    (folder / "config.json").write_text("{}")
    assert cached_pretrained_directory(tmp_path, "lerobot/pi0_base", revision) is None
    (folder / "model.safetensors").write_bytes(b"complete fixture")
    assert cached_pretrained_directory(tmp_path, "lerobot/pi0_base", revision) == folder
    assert cached_pretrained_directory(tmp_path, "lerobot/pi0_base", "main") is None
