import json
from pathlib import Path
import pytest
from yue2 import cli

@pytest.mark.parametrize('identifier', ['../escaped','/absolute','nested/song','.', '..', '', 17, [], True])
def test_invalid_id_never_loads_a_pipeline_or_creates_outputs(tmp_path, monkeypatch, identifier):
    calls = []
    def load(args):
        calls.append(args)
        raise RuntimeError('model loader must not be entered')
    monkeypatch.setattr(cli, 'get_pipe', load)
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps({'id':identifier, 'style':'piano','lyrics':'words'})+'\n')
    output = tmp_path / 'output'
    with pytest.raises(ValueError): cli.main(['batch','--input',str(source),'--output',str(output),'--quiet'])
    assert calls == [] and not output.exists()

def test_duplicate_ids_are_still_rejected_before_loading(tmp_path, monkeypatch):
    source = tmp_path/'input.jsonl'
    source.write_text(''.join(json.dumps({'id':'same','style':'piano','lyrics':'words'})+'\n' for _ in range(2)))
    def load(args): raise AssertionError('model loader must not be entered')
    monkeypatch.setattr(cli,'get_pipe',load)
    with pytest.raises(ValueError, match='unique'): cli.main(['batch','--input',str(source)])
