import json
from pathlib import Path
import pytest
import hashlib
import subprocess
import sys

SCRIPT = Path(__file__).resolve().parents[1]/'skills/yue2-music/scripts/abc_tools.py'
SCORE = ('X:1\r\nT:\r\nM:4/4\r\nL:1/32\r\nQ:1/4=88\r\n'
    'V: Vocal clef=treble name="Vocal Melody" snm="Vocal"\r\n'
    'V: Ins clef=treble name="Ins Melody" snm="Inst."\r\nK:C\r\n'
    '% verse\r\nV: Vocal\r\n"C"C32|\r\nV: Ins\r\nZ|\r\n')

def test_inspect_hashes_original_crlf_bytes(tmp_path):
    source = tmp_path/'source.abc'; source.write_bytes(SCORE.encode())
    result = subprocess.run([sys.executable,str(SCRIPT),'inspect',str(source)],capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)['sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()

def test_strip_chords_preserves_original_line_endings(tmp_path):
    source = tmp_path/'source.abc'; source.write_bytes(SCORE.encode())
    destination = tmp_path/'stripped.abc'
    subprocess.run([sys.executable,str(SCRIPT),'strip-chords',str(source),str(destination)],check=True,capture_output=True)
    assert destination.read_bytes() == SCORE.replace('"C"','').encode()
    result = subprocess.run([sys.executable,str(SCRIPT),'compare',str(source),str(destination)],check=True,capture_output=True,text=True)
    assert json.loads(result.stdout)['match'] is True

def test_strip_chords_disables_windows_output_newline_translation(tmp_path, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('abc_campaign_windows', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    source = tmp_path/'source.abc'; source.write_bytes(SCORE.encode())
    destination = tmp_path/'stripped.abc'
    original_open = Path.open
    def windows_open(path, mode='r', *args, **kwargs):
        # Exercise Python's actual text translation, including the Windows
        # default for omitted newline; explicit newline='' bypasses it.
        if 'b' not in mode and kwargs.get('newline') is None:
            kwargs['newline'] = '\r\n'
        return original_open(path, mode, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', windows_open)
    monkeypatch.setattr(sys, 'argv', [str(SCRIPT),'strip-chords',str(source),str(destination)])
    assert module.main() == 0
    assert destination.read_bytes() == SCORE.replace('"C"','').encode()
