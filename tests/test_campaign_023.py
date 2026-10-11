import pytest
import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from yue2.modeling_yue2 import YuE2ForCausalLM
from test_model import tiny_config

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
