"""Check token attention gradients against a double-precision reference."""

import copy
import os

import pytest

torch = pytest.importorskip("torch")
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

from roboscope.rl.token import RLToken  # noqa: E402


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_token_gradients_match_double_precision(device):
    if device == "cuda" and os.environ.get("ROBOSCOPE_CUDA_TESTS") != "1":
        pytest.skip("GPU regression requires an explicitly selected CUDA device")
    torch.set_num_threads(2)
    torch.manual_seed(17)
    width, layers, heads, batch, length = (960, 2, 8, 16, 128) if device == "cuda" else (32, 1, 4, 2, 9)
    model = RLToken(width, width, layers, heads).to(device).train()
    reference = copy.deepcopy(model).double()
    features = torch.randn(batch, length, width, device=device).bfloat16().float()
    valid = torch.ones(batch, length, dtype=torch.bool, device=device)
    loss = model(features, valid)
    loss.backward()
    with sdpa_kernel(SDPBackend.MATH):
        expected = reference(features, valid)
        expected.backward()
    torch.testing.assert_close(loss.double(), expected, rtol=1e-5, atol=1e-7)
    difference = torch.zeros((), dtype=torch.float64, device=device)
    magnitude = torch.zeros_like(difference)
    for actual, target in zip(model.parameters(), reference.parameters(), strict=True):
        assert torch.isfinite(actual.grad).all()
        difference += (actual.grad.double() - target.grad).square().sum()
        magnitude += target.grad.square().sum()
    assert (difference / magnitude).sqrt().item() < 1e-4
