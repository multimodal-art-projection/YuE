import pytest
import torch
from transformers import Qwen3ForCausalLM
from safetensors.torch import load_file
from yue2.fast import derive_ar_checkpoint
from yue2.modeling_yue2 import YuE2ForCausalLM
from test_model import tiny_config

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
