import json
from pathlib import Path
import pytest
from yue2.storage import collect_hashes, sha256_file

def test_receipt_exclusion_applies_only_to_root_path(tmp_path):
    (tmp_path / 'result.json').write_text('{}')
    nested = tmp_path / 'stages'; nested.mkdir()
    (nested / 'result.json').write_text('nested evidence')
    (tmp_path / 'audio.flac').write_bytes(b'synthetic')
    observed = collect_hashes(tmp_path)
    assert set(observed) == {'stages/result.json', 'audio.flac'}
    assert observed['stages/result.json'] == {'sha256': sha256_file(nested/'result.json'), 'bytes': 15}

def test_explicit_relative_exclusion_does_not_hide_same_basename(tmp_path):
    (tmp_path/'stages').mkdir(); (tmp_path/'other').mkdir()
    (tmp_path/'stages/report.json').write_text('first')
    (tmp_path/'other/report.json').write_text('second')
    assert set(collect_hashes(tmp_path, exclude=('stages/report.json',))) == {'other/report.json'}
