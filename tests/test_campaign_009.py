import json
from pathlib import Path
import pytest
from types import SimpleNamespace
import sys
from yue2 import cli, storage

def test_doctor_resolves_each_requested_revision_without_network(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('YUE2_KIT',str(tmp_path))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:0),backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:False))))
    monkeypatch.setattr(cli.importlib.metadata,'version',lambda package:'synthetic-fixture-version')
    resolved = []
    def resolve(model, **kwargs):
        resolved.append((model, kwargs))
        return tmp_path
    monkeypatch.setattr(storage,'resolve_model',resolve)
    monkeypatch.setattr(storage,'model_identity',lambda path:{'files':{}})
    assert cli.main(['doctor','--verify-hashes','--offline','--revision','model-commit','--vae-revision','vae-commit']) == 0
    report = json.loads(capsys.readouterr().out)
    assert resolved == [('m-a-p/YuE2-3B',{'revision':'model-commit','local_files_only':True}),('m-a-p/YuE2-Vae',{'revision':'vae-commit','local_files_only':True})]
    assert report['validated'] is False
