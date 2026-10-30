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
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM, StaticKVCache
from transformers.cache_utils import StaticCache
from yue2.modeling_yue2 import YuE2ForCausalLM, StaticKVCache
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from yue2.modeling_yue2 import YuE2ForCausalLM


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


def model_cache():
    torch.set_num_threads(1)
    torch.manual_seed(314)
    model = YuE2ForCausalLM(tiny_config()).eval()
    c = model.config
    cache = StaticKVCache(c.num_hidden_layers,1,c.num_key_value_heads,12,c.head_dim,model.dtype,model.device)
    return model, cache

@pytest.mark.parametrize('positions', [[2,3], [5,6], [-1,0]])
@torch.inference_mode()
def test_append_only_cache_refuses_explicit_nonappend_slots_without_mutating(tmp_path, positions):
    model, cache = model_cache()
    model(torch.tensor([[3,5,8,7]]), past_key_values=cache)
    before = [(k.clone(),v.clone()) for k,v in zip(cache.key_cache,cache.value_cache)]
    with pytest.raises(ValueError,match='append'):
        model(torch.tensor([[11,12]]), past_key_values=cache, cache_position=torch.tensor(positions))
    assert cache.get_seq_length() == 4
    assert all(torch.equal(k,oldk) and torch.equal(v,oldv) for (k,v),(oldk,oldv) in
               zip(zip(cache.key_cache,cache.value_cache),before))

@torch.inference_mode()
def test_append_positions_and_hf_rewrites_match_fresh_sequence_oracle():
    model, cache = model_cache()
    original = torch.tensor([[3,5,8,7]])
    replacement = torch.tensor([[11,12]])
    model(original, past_key_values=cache)
    append = model(replacement, past_key_values=cache, cache_position=torch.tensor([4,5])).logits
    expected = model(torch.cat([original,replacement],dim=1),use_cache=False).logits[:, -2:]
    torch.testing.assert_close(append, expected, atol=2e-6,rtol=2e-5)
    hf = StaticCache(config=model.config,max_cache_len=12)
    model(original,past_key_values=hf)
    rewrite = model(replacement,past_key_values=hf,cache_position=torch.tensor([2,3])).logits
    fresh = model(torch.cat([original[:,:2],replacement],dim=1),use_cache=False).logits[:, -2:]
    torch.testing.assert_close(rewrite,fresh,atol=2e-6,rtol=2e-5)


@pytest.mark.parametrize('selector', ['permutation','tail','full-count'])
def test_loss_rejects_logit_selection_before_populating_cache(selector):
    torch.set_num_threads(1)
    model = YuE2ForCausalLM(tiny_config()).eval()
    cache = DynamicCache()
    ids = torch.tensor([[3,5,8]])
    selected = torch.tensor([2,0,1]) if selector == 'permutation' else (1 if selector == 'tail' else 3)
    with pytest.raises(ValueError,match='logits_to_keep=0'):
        model(ids,labels=ids,past_key_values=cache,logits_to_keep=selected)
    assert cache.get_seq_length() == 0

def test_full_causal_loss_matches_shifted_logits_and_backpropagates():
    torch.set_num_threads(1)
    torch.manual_seed(173)
    model = YuE2ForCausalLM(tiny_config()).train()
    ids = torch.tensor([[3,5,8,7]])
    result = model(ids,labels=ids,use_cache=False,logits_to_keep=0)
    expected = F.cross_entropy(result.logits[:,:-1].reshape(-1,71),ids[:,1:].reshape(-1))
    assert torch.equal(result.loss,expected)
    result.loss.backward()
    assert torch.isfinite(model.lm_head.weight.grad).all()
    assert torch.count_nonzero(model.lm_head.weight.grad) > 0
