"""CPU interface/provenance checks; GPU tests are explicitly marked by skips."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file, load_file

from yue2 import fast
from yue2 import quantization as quant
from yue2.protocol import Sampling
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM


def make_checkpoint(path):
    path.mkdir()
    config = YuE2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                       vocab_size=71, max_position_embeddings=64, max_latent_frames=64).to_dict()
    (path / "config.json").write_text(json.dumps(config))
    weights = {name: torch.randn(2, 2, dtype=torch.bfloat16) for name in fast.ar_keys(config)}
    weights["model.layers.0.nar_mlp.up_proj.weight"] = torch.ones(3, 2, dtype=torch.bfloat16)
    save_file(weights, path / "model.safetensors")
    return weights, config


def test_optional_imports_do_not_load_cuda_extensions():
    code = """
import sys, importlib.abc
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'vllm', 'triton'}:
            raise RuntimeError('Unexpected optional import: ' + fullname)
sys.meta_path.insert(0, Deny())
import yue2.fast
import yue2.quantization
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_derived_ar_subset_and_hash_validation(tmp_path):
    original, config = make_checkpoint(tmp_path / "full")
    derived = fast.derive_ar_checkpoint(tmp_path / "full", tmp_path / "cache")
    actual = load_file(derived / "model.safetensors")
    assert set(actual) == fast.ar_keys(config)
    assert all(torch.equal(actual[key], original[key]) for key in actual)
    assert json.loads((derived / "config.json").read_text())["architectures"] == ["Qwen3ForCausalLM"]
    assert fast.derive_ar_checkpoint(tmp_path / "full", tmp_path / "cache") == derived
    (derived / "config.json").write_text('{}')
    with pytest.raises(ValueError, match="Corrupt"):
        fast.derive_ar_checkpoint(tmp_path / "full", tmp_path / "cache")


def test_missing_ar_tensor_fails(tmp_path):
    weights, _ = make_checkpoint(tmp_path / "full")
    del weights['lm_head.weight']
    save_file(weights, tmp_path / "full/model.safetensors")
    with pytest.raises(ValueError, match="Missing AR"):
        fast.derive_ar_checkpoint(tmp_path / "full", tmp_path / "cache")


def test_full_context_kv_budget():
    config = YuE2Config().to_dict()
    assert fast.kv_cache_bytes(config) == int(2.625 * 2**30)
    assert fast.kv_cache_bytes(config, max_sequences=2) == int(5.25 * 2**30)


@pytest.mark.parametrize("cfg,legacy,reason", [(1.01, True, "paired_cfg_uses_torch"),
                                               (1., True, "historical_off_sampling_uses_torch"),
                                               (1., False, "vllm_requires_cuda")])
def test_cfg_and_device_fallback_preserve_protocol(monkeypatch, cfg, legacy, reason):
    from yue2 import sampling
    calls, model = [], object()
    pipe = SimpleNamespace(device=torch.device("cpu"), quantization="none", _load_model=lambda: model)
    prefix, negative = [1, 2], [3]
    sample = Sampling(max_tokens=4, min_tokens=0)
    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        return [9], {"output_tokens": 1}, True
    monkeypatch.setattr(sampling, "generate_tokens", generate)
    result = fast.generate_vllm(pipe, prefix, sample, 71, "semantic", negative=negative,
                               cfg_scale=cfg, legacy_off=legacy)
    assert calls[0][0] == (model, prefix, sample, 71, "semantic")
    assert calls[0][1]['negative'] is negative
    assert calls[0][1]['cfg_scale'] == cfg
    assert calls[0][1]['legacy_off'] == legacy
    assert result[1]['backend_actual'] == 'torch' and result[1]['fallback_reason'] == reason


def test_worker_transport_handles_many_buffered_lines(monkeypatch):
    original_popen = subprocess.Popen
    worker_code = """
import sys,json
json.loads(sys.stdin.readline())
for line in sys.stdin:
    q=json.loads(line)
    lines=['YUE2_FAST\\t'+json.dumps({'event':'token','token':i}) for i in range(100)]
    lines.append('YUE2_FAST\\t'+json.dumps({'event':'result','ids':[12], 'timing':{},'truncated':False}))
    sys.stdout.write('\\n'.join(lines)+'\\n'); sys.stdout.flush()
"""
    monkeypatch.setattr(fast.subprocess, "Popen", lambda command, **kwargs: original_popen([sys.executable, '-c', worker_code], **kwargs))
    pipe = SimpleNamespace(model_dir=Path('.'), device=torch.device('cpu'), memory_budget_gib=24)
    worker = fast._Worker(pipe)
    tokens = []
    try:
        for _ in range(2):
            assert worker.request({'phase': 'abc'}, on_token=lambda phase, token: tokens.append((phase, token)))[0] == [12]
        assert len(tokens) == 200
    finally:
        worker.close()
    assert worker.process.poll() is not None


def test_fp8_tensor_scaling_and_cpu_execution_guard():
    value = torch.tensor([-448., -2., 0., 1., 448.])
    encoded, scale = quant.quantize_tensor(value)
    torch.testing.assert_close(encoded.float() * scale, value)
    layer = quant.FP8Linear(torch.nn.Linear(32, 16, bias=False), "cpu")
    assert layer.weight.dtype == torch.float8_e4m3fn
    with pytest.raises(RuntimeError, match="requires CUDA"):
        layer(torch.randn(1, 32))


def test_fp8_cpu_rejected():
    with pytest.raises(RuntimeError, match="requires CUDA"):
        quant.prepare_fp8_ar(torch.nn.Linear(32, 32), "cpu")


@pytest.mark.skipif(importlib.util.find_spec('vllm') is None, reason='Optional vLLM package is not installed')
def test_window_processor_counting_and_request_moves():
    cls = fast.WindowedPenalty
    processor = cls(SimpleNamespace(scheduler_config=SimpleNamespace(max_num_seqs=2)), 'cpu', False)
    a, b = [2, 2, 3], [1]
    added = [(0, SimpleNamespace(extra_args={fast.PENALTY: 2., fast.WINDOW: 3}), [], a),
             (1, SimpleNamespace(extra_args={fast.PENALTY: 3., fast.WINDOW: 2}), [], b)]
    processor.update_state(SimpleNamespace(removed=[], added=added, moved=[]))
    logits = torch.tensor([[0., 9., 12., -2.], [0., 9., 12., -2.]])
    actual = processor.apply(logits.clone())
    torch.testing.assert_close(actual, torch.tensor([[0., 9., 3., -4.], [0., 3., 12., -2.]]))
    processor.update_state(SimpleNamespace(removed=[], added=[], moved=[(0, 1, SimpleNamespace(name='SWAP'))]))
    torch.testing.assert_close(processor.apply(logits.clone()), actual.flip(0))
    processor.update_state(SimpleNamespace(removed=[0], added=[], moved=[(1, 0, SimpleNamespace(name='UNIDIRECTIONAL'))]))
    torch.testing.assert_close(processor.apply(logits[:1].clone()), actual[:1])


def test_fp8_ar_selection_idempotence_and_exact_restore(monkeypatch):
    config = YuE2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                       vocab_size=71, max_latent_frames=64)
    model = YuE2ForCausalLM(config).to(torch.bfloat16)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    constructor = quant.FP8Linear
    monkeypatch.setattr(quant, 'FP8Linear', lambda linear, device: constructor(linear, 'cpu'))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (9, 0))
    status = quant.prepare_fp8_ar(model, 'cuda')
    assert status['active_ar_linears'] == 7
    assert isinstance(model.model.layers[0].nar_mlp.up_proj, torch.nn.Linear)
    refs = model._yue2_fp8_originals
    assert quant.prepare_fp8_ar(model, 'cuda') == status and model._yue2_fp8_originals is refs
    quant.restore_ar(model)
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in original.items())
    assert quant.quantization_status(model)['active_ar_linears'] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Needs an allocated NVIDIA GPU')
def test_fp8_scaled_mm_cuda():
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip('FP8 kernel requires SM89 or newer')
    torch.manual_seed(17)
    linear = torch.nn.Linear(64, 32, bias=False, dtype=torch.bfloat16, device='cuda')
    layer = quant.FP8Linear(linear, 'cuda')
    for rows in [1, 17]:
        x = torch.randn(rows, 64, dtype=torch.bfloat16, device='cuda')
        actual, expected = layer(x), linear(x)
        assert actual.shape == expected.shape and torch.isfinite(actual).all()
        assert (actual.float() - expected.float()).norm() / expected.float().norm() < .15
