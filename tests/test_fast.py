"""CPU interface/provenance checks; GPU tests are explicitly marked by skips."""
import importlib.util
import json
import os
import numpy as np
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
from yue2.fast import derive_ar_checkpoint
from test_model import tiny_config
from transformers import Qwen3ForCausalLM


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
    # 24576 context plus the 256-token headroom the vLLM admission check
    # needs, rounded to 16-token blocks.
    assert fast.kv_cache_bytes(config) == 24832 * 28 * 2 * 8 * 128 * 2
    assert fast.kv_cache_bytes(config, max_sequences=2) == 2 * 24832 * 28 * 2 * 8 * 128 * 2


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


_v2_sampler_available = fast._v2_sampler_available


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("phase", ["abc", "semantic"])
def test_worker_applies_stage_vocabulary(monkeypatch, tmp_path, v2, phase):
    import asyncio
    import io

    make_checkpoint(tmp_path / "model")
    captured, emitted, stopped = [], [], []

    class Engine:
        @staticmethod
        def from_engine_args(args):
            return Engine()

        async def generate(self, prompt, params, request_id):
            captured.append(params)
            yield SimpleNamespace(outputs=[SimpleNamespace(
                token_ids=[params.stop_token_ids[0]], stop_reason=None)])

        def shutdown(self):
            stopped.append(True)

    def sampling_params(allowed_token_ids=None, **kwargs):
        return SimpleNamespace(allowed_token_ids=allowed_token_ids, **kwargs)

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(
        __version__="0.19.0", SamplingParams=sampling_params))
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", SimpleNamespace(AsyncEngineArgs=SimpleNamespace))
    monkeypatch.setitem(sys.modules, "vllm.v1.engine.async_llm", SimpleNamespace(AsyncLLM=Engine))
    monkeypatch.setenv("YUE2_VLLM_VERSION", "0.19.0")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(total_memory=24 * 2**30))
    monkeypatch.setattr(fast, "derive_ar_checkpoint", lambda path: tmp_path / "model")
    monkeypatch.setattr(fast, "_v2_sampler_available", lambda: v2)
    monkeypatch.setattr(fast, "_emit", emitted.append)
    setup = {"device": "cuda", "model_dir": "unused", "memory_budget_gib": 24}
    request = {"prefix": [1], "sampling": {}, "seed": 71, "phase": phase, "stream_tokens": True}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(setup) + "\n" + json.dumps(request) + "\n"))
    asyncio.run(fast._worker_main())

    params, = captured
    if v2:
        assert params.allowed_token_ids is None
        assert params.extra_args[fast.PHASE] == phase
    else:
        allowed = (list(range(fast.EOD)) + [fast.ABC_END] if phase == "abc" else
                   list(range(fast.CODEC_OFFSET, fast.CODEC_OFFSET + fast.CODEC_SIZE)) + [fast.MUSIC_END])
        assert params.allowed_token_ids == allowed
        assert fast.PHASE not in params.extra_args
    assert [event["event"] for event in emitted] == ["token", "result"]
    assert emitted[-1]["ids"] == [] and not emitted[-1]["truncated"]
    assert stopped == [True]


@pytest.mark.skipif(importlib.util.find_spec('vllm') is None or _v2_sampler_available(),
                    reason='Optional vLLM package or its model-runner V1 sampler is not installed')
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


@pytest.mark.skipif(not _v2_sampler_available(), reason='vLLM model-runner V2 sampler is not installed')
def test_window_processor_v2_reads_committed_tokens():
    cls = fast.WindowedPenalty
    vocab = fast.CODEC_OFFSET + fast.CODEC_SIZE
    req_states = SimpleNamespace(
        device=torch.device("cpu"), max_num_reqs=2, vocab_size=vocab,
        all_token_ids=SimpleNamespace(gpu=torch.tensor([[2, 2, 3], [1, 0, 0]])),
        prompt_len=SimpleNamespace(cpu=torch.tensor([0, 0])),
        prefill_len=SimpleNamespace(cpu=torch.tensor([0, 0])),
        total_len=SimpleNamespace(gpu=torch.tensor([3, 1])))
    processor = cls(SimpleNamespace(), req_states)
    with pytest.raises(ValueError):
        cls.validate_params(SimpleNamespace(extra_args={fast.PENALTY: 0.}))
    assert processor.add_request(0, SimpleNamespace(extra_args={fast.PENALTY: 2., fast.WINDOW: 3}))
    assert processor.add_request(1, SimpleNamespace(extra_args={fast.PENALTY: 3., fast.WINDOW: 2}))
    logits = torch.tensor([[0., 9., 12., -2.], [0., 9., 12., -2.]])
    ctx = SimpleNamespace(idx_mapping_np=np.array([0, 1]))
    actual = processor.apply(logits.clone(), ctx)
    torch.testing.assert_close(actual, torch.tensor([[0., 9., 3., -4.], [0., 3., 12., -2.]]))
    # A recycled slot is fully reinitialized by its next occupant: a neutral
    # penalty disables the adjustment for that row from then on.
    assert not processor.add_request(1, SimpleNamespace(extra_args={fast.PENALTY: 1.}))
    torch.testing.assert_close(processor.apply(logits.clone(), ctx)[1], logits[1])
    # The yue2_phase mask keeps only each stage's range plus its end token.
    assert processor.add_request(0, SimpleNamespace(extra_args={fast.PENALTY: 1., fast.PHASE: "abc"}))
    row = torch.zeros(1, vocab)
    out = processor.apply(row.clone(), SimpleNamespace(idx_mapping_np=np.array([0])))[0]
    assert out[fast.EOD - 1] == 0 and out[fast.ABC_END] == 0
    assert torch.isinf(out[fast.EOD:fast.ABC_END]).all() and torch.isinf(out[fast.ABC_END + 1:]).all()


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


@pytest.mark.parametrize('missing', ['model.safetensors','config.json','all'])
def test_cached_derivative_cannot_bypass_verification_by_omitting_hashes(tmp_path,missing):
    torch.set_num_threads(1)
    source = tmp_path/'source'; YuE2ForCausalLM(tiny_config()).save_pretrained(source)
    cached = derive_ar_checkpoint(source,tmp_path/'cache')
    receipt_path = cached/'derivation.json'; receipt = json.loads(receipt_path.read_text())
    if missing == 'all': receipt['sha256'] = {}
    else: receipt['sha256'].pop(missing)
    receipt_path.write_text(json.dumps(receipt))
    if missing != 'config.json': (cached/'model.safetensors').write_bytes(b'corrupt cache fixture')
    else: (cached/'config.json').write_text('{}')
    with pytest.raises(ValueError,match='manifest'): derive_ar_checkpoint(source,tmp_path/'cache')

def test_complete_cache_keeps_exact_ar_tensors_on_reuse(tmp_path):
    torch.set_num_threads(1); torch.manual_seed(918)
    original = YuE2ForCausalLM(tiny_config())
    source = tmp_path/'source'; original.save_pretrained(source)
    cached = derive_ar_checkpoint(source,tmp_path/'cache')
    assert derive_ar_checkpoint(source,tmp_path/'cache') == cached
    actual = load_file(cached/'model.safetensors')
    assert actual and all(torch.equal(value, original.state_dict()[name]) for name,value in actual.items())


@pytest.mark.parametrize('tied', [True, False])
@torch.inference_mode()
def test_native_tied_checkpoint_saves_reloads_and_derives_equivalent_ar(tmp_path, tied):
    torch.set_num_threads(1)
    torch.manual_seed(94)
    config = tiny_config()
    config.tie_word_embeddings = tied
    model = YuE2ForCausalLM(config).eval()
    assert (model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()) == tied
    source = tmp_path / 'source'
    model.save_pretrained(source)
    native = YuE2ForCausalLM.from_pretrained(source, local_files_only=True).eval()
    ids = torch.tensor([[3,5,8,7]])
    expected = model(ids, use_cache=False).logits
    torch.testing.assert_close(native(ids, use_cache=False).logits, expected, atol=0, rtol=0)
    derived = derive_ar_checkpoint(source, tmp_path / 'cache')
    tensors = load_file(derived / 'model.safetensors')
    assert torch.equal(tensors['lm_head.weight'], model.lm_head.weight)
    assert torch.equal(tensors['model.embed_tokens.weight'], model.model.embed_tokens.weight)
    ar = Qwen3ForCausalLM.from_pretrained(derived, local_files_only=True).eval()
    torch.testing.assert_close(ar(ids, use_cache=False).logits, expected, atol=2e-6, rtol=2e-5)
    assert derive_ar_checkpoint(source, tmp_path / 'cache') == derived


@pytest.mark.parametrize('exit_after_result', [True,False])
def test_worker_drains_short_pipe_reads_before_treating_child_exit_as_failure(monkeypatch, exit_after_result):
    popen = subprocess.Popen
    processes = []
    code = """
import json, sys
json.loads(sys.stdin.readline())
for line in sys.stdin:
    json.loads(line)
    print('YUE2_FAST\t' + json.dumps({'event':'result','ids':list(range(1000)),
                                     'timing':{'fixture':True},'truncated':False}),flush=True)
    if sys.argv[1] == 'exit': break
"""
    def launch(command, **kwargs):
        process = popen([sys.executable,'-c',code,'exit' if exit_after_result else 'persist'],**kwargs)
        processes.append(process)
        return process
    monkeypatch.setattr(fast.subprocess,'Popen',launch)
    native_read = os.read
    def short_read(fd,size):
        if not processes or fd != processes[0].stdout.fileno(): return native_read(fd,size)
        value = native_read(fd,min(size,100))
        if exit_after_result: processes[0].wait(timeout=5)
        return value
    monkeypatch.setattr(fast.os,'read',short_read)
    pipe = SimpleNamespace(model_dir=Path('.'),device='cpu',memory_budget_gib=24)
    worker = fast._Worker(pipe)
    try:
        actual = worker.request({'phase':'abc'})
        assert actual == (list(range(1000)),{'fixture':True},False)
    finally: worker.close()
    assert processes[0].poll() is not None
