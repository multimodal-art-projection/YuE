import json
from pathlib import Path
import pytest
from yue2.storage import model_identity, sha256_file

@pytest.mark.parametrize('size', [0, 999, True, '7'])
def test_inconsistent_weight_manifest_size_is_rejected(tmp_path, size):
    weight = tmp_path / 'model.safetensors'; weight.write_bytes(b'fixture')
    (tmp_path / 'config.json').write_text('{}')
    manifest = {'files': {weight.name: {'sha256': sha256_file(weight), 'bytes': size}}}
    (tmp_path / 'weights_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='integrity'): model_identity(tmp_path)

def test_matching_size_and_hash_are_accepted(tmp_path):
    weight = tmp_path / 'model.safetensors'; weight.write_bytes(b'fixture')
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'weights_manifest.json').write_text(json.dumps({'files': {weight.name: {'sha256': sha256_file(weight), 'bytes': 7}}}))
    assert model_identity(tmp_path)['files'][weight.name]['bytes'] == 7

def test_explicit_verification_opt_out_observes_actual_size(tmp_path):
    weight = tmp_path / 'model.safetensors'; weight.write_bytes(b'fixture')
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'weights_manifest.json').write_text(json.dumps({'files': {weight.name: {'sha256': 'bad', 'bytes': 999}}}))
    assert model_identity(tmp_path, verify=False)['files'][weight.name]['bytes'] == 7
