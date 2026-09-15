"""Small CPU checks of HF APIs; no release checkpoint or GPU required."""
import importlib.util
import json
import os
import struct
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM
from transformers.cache_utils import DynamicCache, StaticCache

from yue2.modeling_yue2 import (
    YuE2Config,
    YuE2ForCausalLM,
    StaticKVCache,
    _mps_broadcast_gqa,
    sdpa,
)

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an Apple MPS device"
)


def tiny_config():
    return YuE2Config(
        hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, intermediate_size=64,
        vocab_size=71, max_position_embeddings=64, max_latent_frames=64,
        pad_token_id=0, eos_token_id=None, bos_token_id=1,
    )


@pytest.fixture
def model():
    torch.manual_seed(41)
    torch.set_num_threads(1)
    return YuE2ForCausalLM(tiny_config()).eval()


def test_mps_decode_buckets_are_bounded(model):
    cache = make_cache(model, "bounded", capacity=9000)
    expected = {1: 2048, 2048: 2048, 2049: 4096, 4096: 4096,
                4097: 8192, 8192: 8192, 8193: 9000, 9000: 9000}
    assert {used: cache._mps_decode_length(used) for used in expected} == expected


def test_mps_decode_bucket_transitions_are_monotonic(model):
    cache = make_cache(model, "bounded", capacity=10_000)
    lengths = [cache._mps_decode_length(used) for used in range(1, 1 + 10_000)]
    assert lengths == sorted(lengths)                    # never shrinks
    assert all(length >= used for used, length in enumerate(lengths, start=1))
    # Only powers of the bucket base (and the capacity cap) may appear.
    allowed = {2048 * 2 ** k for k in range(16)} | {10_000}
    assert set(lengths) <= allowed
    assert lengths[-1] == 10_000                         # capped by capacity


def test_decode_views_are_exact_without_bucketing(model):
    cache = make_cache(model, "bounded", capacity=32)
    cache.MPS_DECODE_BUCKET = 8
    assert not cache._bucket_decode(torch.zeros(1, 2, 1, 8))  # CPU never buckets
    q = torch.zeros(1, 2, 1, 8)
    key = cache.update(q, q, 0)
    assert key[0].shape[2] == 1


@pytest.mark.parametrize("query_heads,kv_heads,length", [(4, 2, 1), (8, 1, 64), (4, 4, 37)])
def test_broadcast_gqa_matches_expanded_reference(query_heads, kv_heads, length):
    """The allocation-free path must equal explicit K/V head duplication."""
    torch.manual_seed(17)
    groups = query_heads // kv_heads
    query = torch.randn(2, query_heads, length, 8)
    key = torch.randn(2, kv_heads, length, 8)
    value = torch.randn_like(key)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key.repeat_interleave(groups, dim=1), value.repeat_interleave(groups, dim=1)
    )
    actual = _mps_broadcast_gqa(query, key, value)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_broadcast_gqa_matches_expanded_reference_is_causal():
    torch.manual_seed(19)
    query = torch.randn(2, 8, 5, 8)
    key = torch.randn(2, 2, 5, 8)
    value = torch.randn_like(key)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query, key.repeat_interleave(4, dim=1), value.repeat_interleave(4, dim=1), is_causal=True
    )
    actual = _mps_broadcast_gqa(query, key, value, is_causal=True)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_bucket_mask_equals_shortened_keys():
    """Hiding unfilled bucket slots must equal attention over the used prefix."""
    torch.manual_seed(23)
    query = torch.randn(1, 4, 1, 8)
    key = torch.randn(1, 2, 16, 8)
    value = torch.randn_like(key)
    for used in (1, 7, 8, 9, 15, 16):
        mask = torch.arange(16)[None, None, None, :] < used
        hidden = _mps_broadcast_gqa(query, key, value, attn_mask=mask)
        exact = _mps_broadcast_gqa(query, key[:, :, :used], value[:, :, :used])
        torch.testing.assert_close(hidden, exact, rtol=2e-5, atol=2e-6)


@torch.inference_mode()
def test_bucketed_decode_matches_full_prefix(model):
    """Emulate MPS bucketing on CPU and cross every bucket boundary."""
    prompt = torch.tensor([[3, 5, 8, 7, 4, 9]])
    steps = [11, 12, 13, 4, 5, 6]
    expected = model(torch.cat([prompt, torch.tensor([steps])], dim=1), use_cache=False).logits
    cache = make_cache(model, "bounded", capacity=len(steps) + prompt.shape[1] + 2)
    cache.MPS_DECODE_BUCKET = 4                       # exercise 4/8/12 transitions
    cache._bucket_decode = lambda key_states: True    # force the MPS decode shape policy
    got = model(prompt, past_key_values=cache).logits
    torch.testing.assert_close(got, expected[:, :prompt.shape[1]], atol=2e-6, rtol=2e-5)
    for i, token in enumerate(steps):
        position = prompt.shape[1] + i
        got = model(torch.tensor([[token]]), past_key_values=cache,
                    cache_position=torch.tensor([position])).logits
        assert cache.key_cache[0].shape[2] >= position + 1  # bucket never truncates
        torch.testing.assert_close(got[:, -1], expected[:, position], atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_padded_decode_keeps_exact_prefix(model):
    """A supplied attention mask must not be paired with a wider bucket."""
    ids = torch.tensor([[3, 5, 8, 7, 4, 9]])
    next_id = torch.tensor([[11]])
    expected = model(torch.cat([ids, next_id], dim=1), use_cache=False).logits
    cache = make_cache(model, "bounded", capacity=16)
    cache.MPS_DECODE_BUCKET = 4                  # would bucket if left unpadded
    cache._bucket_decode = lambda key_states: True
    mask = torch.ones(1, ids.shape[1], dtype=torch.long)
    model(ids, attention_mask=mask, past_key_values=cache)
    next_mask = torch.cat([mask, torch.ones(1, 1, dtype=torch.long)], dim=1)
    got = model(next_id, attention_mask=next_mask, past_key_values=cache).logits
    torch.testing.assert_close(got[:, -1], expected[:, -1], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@requires_mps
def test_mps_grouped_sdpa_matches_expanded_reference(dtype):
    """MPS must not rely on ``enable_gqa`` or SDPA head broadcasting."""
    torch.manual_seed(31)
    device = torch.device("mps")
    query = torch.randn(1, 8, 6, 16, device=device, dtype=dtype)
    key = torch.randn(1, 2, 6, 16, device=device, dtype=dtype)
    value = torch.randn(1, 2, 6, 16, device=device, dtype=dtype)
    reference = torch.nn.functional.scaled_dot_product_attention(
        query.float().cpu(), key.float().cpu().repeat_interleave(4, dim=1),
        value.float().cpu().repeat_interleave(4, dim=1), is_causal=True,
    )
    actual = sdpa(query, key, value, is_causal=True)
    torch.testing.assert_close(actual.float().cpu(), reference, rtol=3e-3, atol=3e-3)


@requires_mps
def test_mps_causal_sdpa_does_not_leak_future():
    """Causal MPS attention must ignore keys that come after the query."""
    torch.manual_seed(37)
    device = torch.device("mps")
    query = torch.randn(1, 2, 12, 16, device=device, dtype=torch.float16)
    key = torch.randn(1, 2, 12, 16, device=device, dtype=torch.float16)
    value = torch.randn(1, 2, 12, 16, device=device, dtype=torch.float16)
    baseline = sdpa(query, key, value, is_causal=True)
    later_key, later_value = key.clone(), value.clone()
    later_key[:, :, -3:] += 5
    later_value[:, :, -3:] += 5
    changed = sdpa(query, later_key, later_value, is_causal=True)
    torch.testing.assert_close(baseline[:, :, :-3], changed[:, :, :-3], rtol=0, atol=0)
    assert not torch.allclose(baseline[:, :, -1], changed[:, :, -1])


@requires_mps
@torch.inference_mode()
def test_mps_bucketed_decode_matches_full_prefix():
    torch.manual_seed(41)
    torch.set_num_threads(1)
    model = YuE2ForCausalLM(tiny_config()).eval()
    device = torch.device("mps")
    prompt = torch.tensor([[3, 5, 8, 7, 4, 9]])
    steps = [11, 12, 13]
    expected = model(torch.cat([prompt, torch.tensor([steps])], dim=1), use_cache=False).logits
    model = model.to(device)
    cache = make_cache(model, "bounded", capacity=len(steps) + prompt.shape[1] + 2)
    cache.MPS_DECODE_BUCKET = 4  # force several bucket transitions at tiny scale
    assert cache._bucket_decode(torch.zeros(1, 2, 1, 8, device=device))
    got = model(prompt.to(device), past_key_values=cache).logits
    torch.testing.assert_close(got.cpu(), expected[:, :prompt.shape[1]], atol=2e-6, rtol=2e-5)
    for i, token in enumerate(steps):
        position = prompt.shape[1] + i
        got = model(torch.tensor([[token]], device=device), past_key_values=cache,
                    cache_position=torch.tensor([position], device=device)).logits
        assert cache.key_cache[0].shape[2] >= position + 1
        torch.testing.assert_close(got[:, -1].cpu(), expected[:, position], atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_inputs_embeds_and_tail_logits(model):
    ids = torch.tensor([[3, 5, 8, 7]])
    expected = model(ids, use_cache=False).logits
    embedded = model(inputs_embeds=model.get_input_embeddings()(ids), use_cache=False).logits
    torch.testing.assert_close(expected, embedded)
    torch.testing.assert_close(model(ids, logits_to_keep=1).logits, expected[:, -1:])
    assert model(ids, return_dict=False, use_cache=False)[0].shape == (1, 4, 71)
    with pytest.raises(ValueError, match="exactly one"):
        model(ids, inputs_embeds=model.get_input_embeddings()(ids))


def make_cache(model, kind, batch=1, capacity=32):
    if kind == "dynamic":
        return DynamicCache()
    if kind == "hf_static":
        return StaticCache(config=model.config, max_cache_len=capacity)
    c = model.config
    return StaticKVCache(c.num_hidden_layers, batch, c.num_key_value_heads,
                         capacity, c.head_dim, model.dtype, model.device)


@pytest.mark.parametrize("kind", ["dynamic", "hf_static", "bounded"])
@torch.inference_mode()
def test_cached_chunks_equal_full(model, kind):
    ids = torch.tensor([[3, 5, 8, 7, 4, 9]])
    expected = model(ids, use_cache=False).logits
    cache = make_cache(model, kind)
    outputs = []
    for start, end in [(0, 2), (2, 5), (5, 6)]:
        outputs.append(model(ids[:, start:end], past_key_values=cache,
                             cache_position=torch.arange(start, end)).logits)
    torch.testing.assert_close(torch.cat(outputs, dim=1), expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("kind", ["dynamic", "hf_static", "bounded"])
@torch.inference_mode()
def test_left_padding_and_decode(model, kind):
    ids = torch.tensor([[0, 0, 3, 5], [8, 7, 4, 9]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    cache = make_cache(model, kind, batch=2)
    batched = model(ids, attention_mask=mask, past_key_values=cache).logits
    for i, start in enumerate([2, 0]):
        expected = model(ids[i:i+1, start:], use_cache=False).logits
        torch.testing.assert_close(batched[i:i+1, start:], expected, atol=2e-6, rtol=2e-5)
    next_ids = torch.tensor([[11], [12]])
    next_mask = torch.cat([mask, torch.ones(2, 1, dtype=mask.dtype)], dim=1)
    actual = model(next_ids, attention_mask=next_mask, past_key_values=cache).logits
    for i, start in enumerate([2, 0]):
        unpadded = torch.cat([ids[i:i+1, start:], next_ids[i:i+1]], dim=1)
        torch.testing.assert_close(actual[i:i+1], model(unpadded, use_cache=False).logits[:, -1:], atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("cache_implementation", ["dynamic", "static"])
@torch.inference_mode()
def test_hf_generate_padding_and_embeddings(model, cache_implementation):
    ids = torch.tensor([[0, 0, 3, 5], [8, 7, 4, 9]])
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 1, 1]])
    args = dict(max_new_tokens=3, do_sample=False, cache_implementation=cache_implementation)
    outputs = model.generate(ids, attention_mask=mask, **args)
    for i, start in enumerate([2, 0]):
        alone = model.generate(ids[i:i+1, start:], **args)
        torch.testing.assert_close(outputs[i, -3:], alone[0, -3:])
    embedded = model.generate(inputs_embeds=model.get_input_embeddings()(ids), attention_mask=mask, **args)
    torch.testing.assert_close(outputs[:, -3:], embedded[:, -3:])


@torch.inference_mode()
def test_explicit_position_ids(model):
    ids = torch.tensor([[3, 5, 8, 7]])
    positions = torch.tensor([[1, 4, 9, 10]])
    expected = model(ids, position_ids=positions, use_cache=False).logits
    cache = DynamicCache()
    first = model(ids[:, :3], position_ids=positions[:, :3], past_key_values=cache).logits
    last = model(ids[:, 3:], position_ids=positions[:, 3:], past_key_values=cache).logits
    torch.testing.assert_close(torch.cat([first, last], dim=1), expected, atol=2e-6, rtol=2e-5)


@torch.inference_mode()
def test_save_load_and_remote_auto_class(model, tmp_path):
    model.save_pretrained(tmp_path, safe_serialization=True)
    ids = torch.tensor([[3, 5, 8]])
    expected = model(ids, use_cache=False).logits
    loaded = YuE2ForCausalLM.from_pretrained(tmp_path, local_files_only=True)
    torch.testing.assert_close(expected, loaded(ids, use_cache=False).logits)
    automatic = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True, local_files_only=True)
    torch.testing.assert_close(expected, automatic(ids, use_cache=False).logits)


@torch.inference_mode()
def test_bounded_cache_overflow_and_reset(model):
    cache = make_cache(model, "bounded", capacity=3)
    model(torch.tensor([[3, 5, 8]]), past_key_values=cache)
    with pytest.raises(ValueError, match="capacity"):
        model(torch.tensor([[7]]), past_key_values=cache)
    cache.reset()
    expected = model(torch.tensor([[9]]), use_cache=False).logits
    torch.testing.assert_close(model(torch.tensor([[9]]), past_key_values=cache).logits, expected)


@torch.inference_mode()
def test_training_labels_and_causality(model):
    first = torch.tensor([[3, 5, 8, 7]])
    second = torch.tensor([[3, 5, 12, 19]])
    torch.testing.assert_close(model(first, use_cache=False).logits[:, :2], model(second, use_cache=False).logits[:, :2])
    assert torch.isfinite(model(first, labels=first).loss)
    with pytest.raises(ValueError, match="Loss"):
        model(first, labels=first, logits_to_keep=1)


def test_release_tensor_names_and_shapes_when_source_available():
    # This audit is optional for a downloaded release: no original cluster path
    # is a runtime dependency. Production builds run it before packaging.
    location = os.environ.get('YUE2_AUDIT_CHECKPOINT')
    if not location:
        pytest.skip('Original release is not installed; use packaged SHA256 manifest')
    source = Path(location)
    with source.open('rb') as stream:
        length = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(length))
    with torch.device('meta'):
        actual = YuE2ForCausalLM(YuE2Config()).state_dict()
    expected = {key: item['shape'] for key, item in header.items() if key != '__metadata__'}
    assert {key: list(tensor.shape) for key, tensor in actual.items()} == expected
