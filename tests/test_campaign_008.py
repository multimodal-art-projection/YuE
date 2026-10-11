import json
from pathlib import Path
import pytest
import importlib.util
import sys
SCRIPT = Path(__file__).resolve().parents[1]/'skills/yue2-music/scripts/listen.py'
spec = importlib.util.spec_from_file_location('listen_campaign', SCRIPT)
listen = importlib.util.module_from_spec(spec); spec.loader.exec_module(listen)
def test_metadata_changed_after_validation_is_not_exported(tmp_path, monkeypatch):
    source = tmp_path/'song'; source.mkdir()
    original = b'{ "style": "piano", "lyrics": "words", "id": "song" }\n'
    request = source/'request.json'; request.write_bytes(original)
    read = listen.json_object
    def read_then_change(path, *args, **kwargs):
        result = read(path, *args, **kwargs)
        if path.name == 'request.json':
            path.write_text(json.dumps({'token':'SYNTHETIC_TEST_SECRET_ONLY'}))
        return result
    monkeypatch.setattr(listen,'json_object',read_then_change)
    output = tmp_path/'page'
    manifest = listen.build([source],output)
    assert (output/'case-001/request.json').read_bytes() == original
    assert manifest['cases'][0]['request']['style'] == 'piano'
    for path in output.rglob('*'):
        if path.is_file(): assert b'SYNTHETIC_TEST_SECRET_ONLY' not in path.read_bytes()

def test_metadata_snapshot_preserves_exact_utf8_bytes(tmp_path):
    source = tmp_path/'song'; source.mkdir()
    original = '{ "style": "钢琴", "lyrics": "words", "id": "song" }\r\n'.encode()
    (source/'request.json').write_bytes(original)
    output = tmp_path/'page'; listen.build([source],output)
    assert (output/'case-001/request.json').read_bytes() == original
