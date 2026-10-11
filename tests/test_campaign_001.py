import json
from pathlib import Path
import pytest
import threading
from yue2 import storage

def test_overlapping_writes_use_independent_temporary_files(tmp_path, monkeypatch):
    target = tmp_path / 'result.json'
    first_waiting, release = threading.Event(), threading.Event()
    original = storage.os.replace
    errors = []
    def replace(source, destination):
        if threading.current_thread().name == 'first-writer':
            first_waiting.set()
            assert release.wait(5)
        return original(source, destination)
    monkeypatch.setattr(storage.os, 'replace', replace)
    def first():
        try: storage.write_json(target, {'writer': 'first'})
        except BaseException as error: errors.append(error)
    thread = threading.Thread(target=first, name='first-writer')
    thread.start()
    try:
        assert first_waiting.wait(5)
        storage.write_json(target, {'writer': 'second'})
        assert json.loads(target.read_text()) == {'writer': 'second'}
    finally:
        release.set(); thread.join(5)
    assert not thread.is_alive()
    assert errors == []
    assert json.loads(target.read_text()) == {'writer': 'first'}
    assert list(tmp_path.iterdir()) == [target]

def test_failed_serialization_leaves_previous_receipt_intact(tmp_path):
    target = tmp_path / 'result.json'
    storage.write_json(target, {'complete': True})
    with pytest.raises(ValueError): storage.write_json(target, {'value': float('nan')})
    assert json.loads(target.read_text()) == {'complete': True}
    assert list(tmp_path.iterdir()) == [target]

def test_replace_failure_removes_temporary_file(tmp_path, monkeypatch):
    target = tmp_path / 'result.json'
    def fail(*args): raise OSError('synthetic filesystem boundary failure')
    monkeypatch.setattr(storage.os, 'replace', fail)
    with pytest.raises(OSError): storage.write_json(target, {'complete': True})
    assert list(tmp_path.iterdir()) == []
