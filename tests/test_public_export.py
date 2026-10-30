"""Public export rejects directory residue and retains usable HF models."""
import json
from pathlib import Path
import pytest
import torch
from transformers import AutoModelForCausalLM
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from yue2.modeling_vae import YuE2VAEConfig
from yue2.storage import copy_model_files, resolve_model
from test_model import tiny_config
from transformers import AutoModel, AutoModelForCausalLM
from yue2.modeling_yue2 import YuE2ForCausalLM
from yue2.modeling_vae import YuE2VAE
from yue2.storage import copy_model_files
from test_model import tiny_config as ar_config
from test_vae import tiny_config as vae_config


@pytest.mark.parametrize('config_class', [YuE2Config, YuE2VAEConfig])
def test_unknown_metadata_is_not_loaded_or_reserialized(config_class):
    config = config_class(unused_option={'private_note': 'DO_NOT_EXPORT'},
                          _name_or_path='/private/build/source')
    config.extra_note = 'DO_NOT_EXPORT'
    text = json.dumps(config.to_dict())
    assert 'DO_NOT_EXPORT' not in text and '/private/build' not in text
    assert not hasattr(config, 'unused_option')


def test_model_export_omits_unlisted_code_reports_and_cache(tmp_path):
    torch.set_num_threads(1)
    source, exported = tmp_path / 'source', tmp_path / 'export'
    model = YuE2ForCausalLM(tiny_config()).eval()
    model.save_pretrained(source)
    (source / 'notes.json').write_text('{"private_note":"DO_NOT_EXPORT"}')
    (source / 'extra_model.py').write_text('DO_NOT_EXPORT')
    (source / 'modeling_yue2.py').write_text('DO_NOT_EXPORT')
    (source / '.cache').mkdir()
    (source / '.cache/report.txt').write_text('DO_NOT_EXPORT')
    config = json.loads((source / 'config.json').read_text())
    config['unused_option'] = 'DO_NOT_EXPORT'
    (source / 'config.json').write_text(json.dumps(config))
    copy_model_files(source, exported)
    assert not (exported / 'notes.json').exists()
    assert not (exported / 'extra_model.py').exists()
    assert not (exported / '.cache').exists()
    assert 'DO_NOT_EXPORT' not in (exported / 'config.json').read_text(encoding='utf-8')
    assert 'DO_NOT_EXPORT' not in (exported / 'modeling_yue2.py').read_text(encoding='utf-8')
    restored = AutoModelForCausalLM.from_pretrained(exported, trust_remote_code=True).eval()
    ids = torch.tensor([[1, 5, 3]])
    with torch.inference_mode():
        torch.testing.assert_close(model(ids).logits, restored(ids).logits, rtol=0, atol=0)


def test_export_refuses_nonempty_destination(tmp_path):
    destination = tmp_path / 'output'; destination.mkdir()
    (destination / 'old_report.txt').write_text('DO_NOT_EXPORT')
    with pytest.raises(FileExistsError):
        copy_model_files(tmp_path / 'source', destination)


def test_pipeline_export_refuses_residue_at_the_root(tmp_path):
    from yue2.pipeline import YuE2Pipeline
    pipe = YuE2Pipeline.__new__(YuE2Pipeline)
    (tmp_path / 'old_report.txt').write_text('DO_NOT_EXPORT')
    with pytest.raises(FileExistsError, match='empty pipeline'):
        pipe.save_pretrained(tmp_path)


def test_hf_download_is_restricted_to_inference_files(monkeypatch, tmp_path):
    import huggingface_hub
    captured = {}
    def download(*args, **kwargs):
        captured.update(kwargs)
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    resolve_model('example/YuE2', revision='a' * 40, local_files_only=True)
    assert '*.json' not in captured['allow_patterns'] and '*.py' not in captured['allow_patterns']
    assert 'config.json' in captured['allow_patterns']
    assert captured['revision'] == 'a' * 40 and captured['local_files_only']


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
