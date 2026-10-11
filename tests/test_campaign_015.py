import json
import pytest
import torch
from transformers import AutoModel, AutoModelForCausalLM
from yue2.modeling_yue2 import YuE2ForCausalLM
from yue2.modeling_vae import YuE2VAE
from yue2.storage import copy_model_files
from test_model import tiny_config as ar_config
from test_vae import tiny_config as vae_config

@pytest.mark.parametrize('kind', ['yue2', 'yue2_vae'])
@pytest.mark.parametrize('redirect', [False, True])
@torch.inference_mode()
def test_export_auto_classes_resolve_to_exported_reviewed_code(tmp_path, kind, redirect):
    torch.set_num_threads(1)
    torch.manual_seed(241)
    model = (YuE2ForCausalLM(ar_config()) if kind == 'yue2' else YuE2VAE(vae_config())).eval()
    source = tmp_path / 'source'
    model.save_pretrained(source)
    if redirect:
        config_path = source / 'config.json'
        config = json.loads(config_path.read_text())
        config['auto_map'] = {'AutoConfig': 'foreign_config.RemoteConfig',
                              'AutoModelForCausalLM': 'foreign_model.RemoteAR',
                              'AutoModel': 'foreign_model.RemoteVAE'}
        config_path.write_text(json.dumps(config))
    exported = tmp_path / 'exported'
    copy_model_files(source, exported)
    mapping = json.loads((exported / 'config.json').read_text())['auto_map']
    expected = ({'AutoConfig': 'modeling_yue2.YuE2Config',
                 'AutoModelForCausalLM': 'modeling_yue2.YuE2ForCausalLM'} if kind == 'yue2' else
                {'AutoConfig': 'modeling_vae.YuE2VAEConfig', 'AutoModel': 'modeling_vae.YuE2VAE'})
    assert mapping == expected
    if kind == 'yue2':
        restored = AutoModelForCausalLM.from_pretrained(exported, trust_remote_code=True, local_files_only=True).eval()
        ids = torch.tensor([[3, 5, 8]])
        torch.testing.assert_close(restored(ids, use_cache=False).logits, model(ids, use_cache=False).logits,
                                   rtol=0, atol=0)
    else:
        restored = AutoModel.from_pretrained(exported, trust_remote_code=True, local_files_only=True).eval()
        latent = torch.randn(1, 2, 3)
        torch.testing.assert_close(restored.decode(latent), model.decode(latent), rtol=0, atol=0)
