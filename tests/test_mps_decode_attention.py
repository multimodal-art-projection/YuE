"""MPS decode attention folds grouped query heads instead of copying the KV cache."""
import pytest
import torch
import torch.nn.functional as F

from yue2.modeling_yue2 import sdpa

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")


def reference(q, k, v, mask):
    groups = q.shape[1] // k.shape[1]
    if mask is not None and mask.dtype != torch.bool:
        mask = mask.float()
    return F.scaled_dot_product_attention(q.float(), k.float().repeat_interleave(groups, 1),
                                          v.float().repeat_interleave(groups, 1), attn_mask=mask)


@pytest.mark.parametrize("keys", [17, 700, 3000])
@pytest.mark.parametrize("mask", [None, "bool", "float"])
def test_folded_decode_attention_matches_fp32(keys, mask):
    generator = torch.Generator().manual_seed(keys)
    shape = (1, 8, keys, 128)
    q = torch.randn(1, 16, 1, 128, generator=generator).bfloat16().to("mps")
    k = torch.randn(shape, generator=generator).bfloat16().to("mps")
    v = torch.randn(shape, generator=generator).bfloat16().to("mps")
    attn_mask = None
    if mask is not None:
        visible = torch.rand(1, 1, 1, keys, generator=generator) > 0.1
        visible[..., 0] = True
        attn_mask = visible if mask == "bool" else torch.zeros(visible.shape).masked_fill(~visible, float("-inf")).bfloat16()
        attn_mask = attn_mask.to("mps")
    # PyTorch 2.10 returns wrong values when the first MPS attention call in a process takes
    # its two-pass kernel (1024+ keys); a short call first avoids that unrelated bug.
    F.scaled_dot_product_attention(q, k[:, :, :16].repeat_interleave(2, 1), v[:, :, :16].repeat_interleave(2, 1))
    result = sdpa(q, k, v, attn_mask=attn_mask)
    expected = reference(q, k, v, attn_mask)
    torch.testing.assert_close(result.float(), expected, atol=2e-2, rtol=2e-2)


def test_multi_row_queries_keep_the_repeat_path(monkeypatch):
    q = torch.randn(1, 16, 5, 128).bfloat16().to("mps")
    k = torch.randn(1, 8, 40, 128).bfloat16().to("mps")
    calls = []
    real = F.scaled_dot_product_attention

    def spy(query, key, *args, **kwargs):
        calls.append((query.shape, key.shape))
        return real(query, key, *args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    sdpa(q, k, k)
    assert calls == [(torch.Size([1, 16, 5, 128]), torch.Size([1, 16, 40, 128]))]


def test_decode_attends_without_copying_the_kv_cache(monkeypatch):
    q = torch.randn(1, 16, 1, 128).bfloat16().to("mps")
    k = torch.randn(1, 8, 40, 128).bfloat16().to("mps")
    calls = []
    real = F.scaled_dot_product_attention

    def spy(query, key, *args, **kwargs):
        calls.append((query.shape, key.shape))
        return real(query, key, *args, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    sdpa(q, k, k)
    assert calls == [(torch.Size([1, 8, 2, 128]), torch.Size([1, 8, 40, 128]))]
