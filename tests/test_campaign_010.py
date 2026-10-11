import json
from pathlib import Path
import pytest
import torch
from huggingface_hub.file_download import _create_symlink
from yue2.modeling_vae import YuE2VAE
from test_vae import tiny_config

@pytest.mark.parametrize('decoder_only', [False,True])
@pytest.mark.parametrize('mode', ['parent','absolute','nested-parent'])
def test_indexed_shard_rejects_absolute_and_parent_filenames(tmp_path, mode, decoder_only):
    torch.set_num_threads(1)
    directory = tmp_path/'snapshot'
    original = YuE2VAE(tiny_config()); original.save_pretrained(directory)
    outside = tmp_path/'outside.safetensors'
    (directory/'model.safetensors').replace(outside)
    if mode == 'parent': name = '../outside.safetensors'
    elif mode == 'absolute': name = str(outside)
    else:
        (directory/'shards').mkdir()
        name = 'shards/../../outside.safetensors'
    (directory/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{key:name for key in original.state_dict()}}))
    with pytest.raises(ValueError,match='shard path'):
        YuE2VAE.from_pretrained(directory,decoder_only=decoder_only,local_files_only=True)

@pytest.mark.parametrize('decoder_only', [False,True])
def test_safe_nested_indexed_snapshot_preserves_exact_decoder_output(tmp_path,decoder_only):
    torch.set_num_threads(1); torch.manual_seed(247)
    directory = tmp_path/'snapshot'
    original = YuE2VAE(tiny_config()); original.save_pretrained(directory)
    (directory/'shards').mkdir()
    (directory/'model.safetensors').replace(directory/'shards/weights.safetensors')
    (directory/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{key:'shards/weights.safetensors' for key in original.state_dict()}}))
    loaded = YuE2VAE.from_pretrained(directory,decoder_only=decoder_only,local_files_only=True)
    latent = torch.randn(1,2,3)
    assert torch.equal(original.decode(latent), loaded.decode(latent))


@pytest.mark.parametrize('decoder_only', [False, True])
def test_standard_hf_cache_snapshot_can_load_indexed_vae(tmp_path, decoder_only):
    torch.set_num_threads(1)
    torch.manual_seed(247)
    repo_cache = tmp_path/'models--owner--vae'
    snapshot = repo_cache/'snapshots'/'abc123'
    model = YuE2VAE(tiny_config())
    model.save_pretrained(snapshot)
    blobs = repo_cache/'blobs'; blobs.mkdir()
    blob = blobs/'weights-content-hash'
    shard = snapshot/'model-00001-of-00001.safetensors'
    (snapshot/'model.safetensors').replace(blob)
    _create_symlink(str(blob), str(shard), new_blob=False)
    (snapshot/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {name: shard.name for name in model.state_dict()}}))
    assert shard.is_symlink()
    loaded = YuE2VAE.from_pretrained(snapshot, decoder_only=decoder_only, local_files_only=True)
    latent = torch.randn(1,2,3)
    assert torch.equal(model.decode(latent), loaded.decode(latent))

@pytest.mark.parametrize('decoder_only', [False, True])
@pytest.mark.parametrize('name', [r'\outside.safetensors', r'C:outside.safetensors', r'..\outside.safetensors'])
def test_portable_indexed_shard_filenames_reject_windows_root_drive_and_parent(tmp_path, name, decoder_only):
    if Path(name).name != name:
        pytest.skip('POSIX literal-backslash fixture exercises portable filename validation')
    torch.set_num_threads(1)
    directory = tmp_path/'snapshot'
    original = YuE2VAE(tiny_config()); original.save_pretrained(directory)
    (directory/'model.safetensors').replace(directory/name)
    (directory/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{key:name for key in original.state_dict()}}))
    with pytest.raises(ValueError,match='shard path'):
        YuE2VAE.from_pretrained(directory,decoder_only=decoder_only,local_files_only=True)
