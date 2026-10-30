from types import SimpleNamespace

import pytest
import torch

from yue2 import pipeline, placement, quantization
from yue2.modeling_yue2 import YuE2ForCausalLM
from test_model import tiny_config


def test_phase_switch_evicts_all_inactive_modules_before_loading(monkeypatch):
    events = []
    class Module:
        def __init__(self, name):
            self.name = name
        def to(self, device):
            events.append((self.name, str(device)))
    ar = [Module("ar") for _ in range(2)]
    nar = [Module("nar") for _ in range(2)]
    model = SimpleNamespace(model=SimpleNamespace(norm=Module("shared"), rotary_emb=Module("shared")))
    monkeypatch.setattr(placement, "ar_modules", lambda _: iter(ar))
    monkeypatch.setattr(placement, "nar_modules", lambda _: iter(nar))
    monkeypatch.setattr(placement, "empty_cache", lambda _: events.append(("empty", "")))
    placement.activate_phase(model, "nar", "xpu")
    assert events == [("ar", "cpu"), ("ar", "cpu"), ("empty", ""),
                      ("nar", "xpu"), ("nar", "xpu"), ("shared", "xpu"), ("shared", "xpu")]


def test_module_groups_cover_every_parameter_and_registered_buffer():
    model = YuE2ForCausalLM(tiny_config()).eval()
    groups = [*placement.ar_modules(model), *placement.nar_modules(model), model.model.norm, model.model.rotary_emb]
    covered = [value for group in groups for value in [*group.parameters(), *group.buffers()]]
    assert {id(x) for x in covered} == {id(x) for x in [*model.parameters(), *model.buffers()]}
    assert len(covered) == len({id(x) for x in covered})


def test_nar_load_restores_original_bf16_without_preparing_fp8(monkeypatch):
    model = YuE2ForCausalLM(tiny_config()).bfloat16().eval()
    name = "model.layers.0.self_attn.q_proj"
    original = model.get_submodule(name)
    quantization._replace(model, name, quantization.FP8Linear(original, "cpu"))
    object.__setattr__(model, "_yue2_fp8_originals", {name: original})
    pipe = pipeline.YuE2Pipeline.__new__(pipeline.YuE2Pipeline)
    pipe.device, pipe.progress, pipe.low_vram = torch.device("cpu"), False, True
    pipe._model, pipe.quantization = model, "fp8"
    monkeypatch.setattr(quantization, "prepare_fp8_ar", lambda *a: pytest.fail("NAR re-enabled FP8"))
    assert pipe._load_model(for_nar=True) is model
    assert model.get_submodule(name) is original
    assert not model._yue2_fp8_originals
    assert all(p.dtype == torch.bfloat16 for p in model.parameters())


@pytest.mark.parametrize("budget", [6, 8, 12, 24, 48])
def test_low_vram_respects_cuda_memory_budget(tmp_path, monkeypatch, budget):
    fractions = []
    monkeypatch.setattr(pipeline, "resolve_device", lambda _: torch.device("cuda:0"))
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=24 * 2**30))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda value, device: fractions.append(value))
    monkeypatch.setattr(pipeline, "YuE2TextTokenizer", lambda _: None)
    monkeypatch.setattr(pipeline, "model_identity", lambda *args: {})
    pipeline.YuE2Pipeline(tmp_path, tmp_path, memory_budget_gib=budget, low_vram=True, progress=False)
    assert fractions == [min(budget - 2, 22) / 24]
