from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from yue2.cuda_graph import attention_capabilities, GraphAR
from yue2.modeling_yue2 import _folded_gqa, sdpa, YuE2ForCausalLM, StaticKVCache
from test_model import tiny_config


@pytest.mark.parametrize("rows", [1, 2, 4])
@pytest.mark.parametrize("mask_kind", [None, "bool", "float"])
def test_query_folding_matches_gqa_without_expanding_kv(rows, mask_kind, monkeypatch):
    torch.manual_seed(12)
    q, k, v = torch.randn(2, 8, rows, 8), torch.randn(2, 2, 19, 8), torch.randn(2, 2, 19, 8)
    mask = None
    if mask_kind:
        mask = torch.rand(2, 1, rows, 19) > .2
        mask[..., 0] = True
        if mask_kind == "float":
            mask = torch.zeros(mask.shape).masked_fill(~mask, -torch.inf)
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, enable_gqa=True)
    original, pointers = F.scaled_dot_product_attention, []
    def observe(query, key, value, **kwargs):
        pointers.append((key.data_ptr(), value.data_ptr()))
        return original(query, key, value, **kwargs)
    monkeypatch.setattr(F, "scaled_dot_product_attention", observe)
    torch.testing.assert_close(_folded_gqa(q, k, v, mask), expected, rtol=2e-5, atol=2e-6)
    assert pointers == [(k.data_ptr(), v.data_ptr())]


def test_bucket_boundaries_and_reset_hide_unwritten_slots(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(18)
    model = YuE2ForCausalLM(tiny_config()).eval()
    config = model.config
    cache = StaticKVCache(config.num_hidden_layers, 1, config.num_key_value_heads, 19,
                          config.head_dim, torch.float32, torch.device("cpu"))
    monkeypatch.setattr(cache, "_bucket_decode", lambda _: True)
    monkeypatch.setattr(cache, "MPS_DECODE_BUCKET", 4)
    assert [cache._mps_decode_length(n) for n in [1, 4, 5, 8, 9, 16, 17, 19]] == [4, 4, 8, 8, 16, 16, 19, 19]
    with torch.inference_mode():
        for count in [19, 9]:
            cache.reset()
            for value in cache.key_cache + cache.value_cache:
                value.fill_(1000)
            ids = torch.randint(0, config.vocab_size, (1, count))
            expected = model(ids, use_cache=False).logits
            outputs = [model(ids[:, i:i+1], past_key_values=cache).logits for i in range(count)]
            torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("hip", [None, "7.1"])
def test_capabilities_reject_missing_kernels_and_rocm_private_ops(monkeypatch, hip):
    config = tiny_config()
    weight = SimpleNamespace(device=torch.device("cuda:0"), dtype=torch.bfloat16,
                             new_zeros=lambda shape: torch.zeros(shape, dtype=torch.bfloat16))
    model = SimpleNamespace(config=config, model=SimpleNamespace(embed_tokens=SimpleNamespace(weight=weight)))
    monkeypatch.setattr(torch.version, "hip", hip)
    monkeypatch.setattr(torch.backends.cuda, "is_flash_attention_available", lambda: False)
    monkeypatch.setattr(torch.backends.cuda, "can_use_flash_attention", lambda *a: pytest.fail("stub probed"))
    monkeypatch.setattr(torch.backends.cudnn, "is_available", lambda: False)
    assert attention_capabilities(model) == {"flash": False, "cudnn": False, "sdpa": True}


def test_cudnn_head_expansion_preserves_group_order(monkeypatch):
    torch.set_num_threads(1)
    model = YuE2ForCausalLM(tiny_config()).eval()
    with torch.inference_mode():
        graph = GraphAR(model, [[1, 2], [3]], 3, capture=False)
        graph.prefill()
        expected = graph.step(4).clone()
        graph.close()
        graph = GraphAR(model, [[1, 2], [3]], 3, capture=False)
        graph.prefill()
        graph.attention_backend = "cudnn"
        monkeypatch.setattr(torch.nn.attention, "sdpa_kernel", lambda *a, **k: nullcontext())
        torch.testing.assert_close(graph.step(4), expected, rtol=2e-5, atol=2e-6)
        graph.close()


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_mps_causal_prefill_has_no_future_leak(dtype):
    torch.manual_seed(9)
    q, k, v = [torch.randn(1, h, 17, 64).to("mps", dtype=dtype) for h in (4, 2, 2)]
    expected = F.scaled_dot_product_attention(q.float().cpu(), k.float().cpu(), v.float().cpu(),
                                             is_causal=True, enable_gqa=True)
    actual = sdpa(q, k, v, is_causal=True).float().cpu()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
    k[..., 8:, :] += 10
    v[..., 8:, :] -= 20
    changed = sdpa(q, k, v, is_causal=True).float().cpu()
    torch.testing.assert_close(changed[..., :8, :], actual[..., :8, :], atol=0, rtol=0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires Apple MPS")
def test_mps_long_decode_from_cold_process():
    import os
    import subprocess
    import sys
    code = '''
import torch
import torch.nn.functional as F
from yue2.modeling_yue2 import sdpa
torch.manual_seed(3)
q = torch.randn(1, 16, 1, 128).bfloat16()
k, v = [torch.randn(1, 8, 3000, 128).bfloat16() for _ in range(2)]
expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), enable_gqa=True)
actual = sdpa(q.to("mps"), k.to("mps"), v.to("mps")).float().cpu()
torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
'''
    subprocess.run([sys.executable, "-c", code], env=dict(os.environ), check=True, timeout=120)
