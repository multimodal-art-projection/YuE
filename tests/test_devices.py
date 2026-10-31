from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from yue2 import devices


def test_explicit_xpu_selection_synchronization_and_cleanup(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.xpu, "is_available", lambda: True)
    monkeypatch.setattr(torch.xpu, "current_device", lambda: 1)
    monkeypatch.setattr(torch.xpu, "synchronize", lambda device: calls.append(("sync", str(device))))
    monkeypatch.setattr(torch.xpu, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.xpu, "empty_cache", lambda: calls.append(("empty",)))
    device = devices.resolve_device("xpu")
    assert str(device) == "xpu:1"
    devices.synchronize(device)
    devices.empty_cache(device)
    assert calls == [("sync", "xpu:1"), ("empty",)]


def test_unavailable_xpu_fails_before_model_loading(monkeypatch):
    monkeypatch.setattr(torch.xpu, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        devices.resolve_device("xpu")


@pytest.mark.skipif(not torch.xpu.is_available(), reason="requires Intel XPU")
def test_xpu_request_rng_is_local_and_repeatable():
    before = torch.xpu.get_rng_state().clone()
    probability = torch.tensor([0.2, 0.8], device="xpu")
    def draw():
        rng = torch.Generator(device="xpu").manual_seed(42)
        return torch.multinomial(probability, 32, replacement=True, generator=rng)
    assert torch.equal(draw(), draw())
    assert torch.equal(before, torch.xpu.get_rng_state())
