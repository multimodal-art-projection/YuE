import json
from pathlib import Path
import pytest
import torch
from yue2.fast import derive_ar_checkpoint
from yue2.modeling_yue2 import YuE2ForCausalLM
from safetensors.torch import load_file
from test_model import tiny_config

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
