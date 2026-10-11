import pytest
import torch
from transformers.cache_utils import StaticCache
from yue2.modeling_yue2 import YuE2ForCausalLM, StaticKVCache
from test_model import tiny_config

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
