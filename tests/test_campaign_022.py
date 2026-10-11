import os
from pathlib import Path
import subprocess
import sys
import zipfile

def test_skill_only_builder_needs_no_optional_toml_parser(tmp_path):
    root = Path(__file__).resolve().parents[1]
    script = root / 'tools/build_release.py'
    code = """
import importlib.abc, runpy, sys
class MissingTomllib(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'tomllib': raise ModuleNotFoundError('tomllib is absent on Python 3.10')
sys.meta_path.insert(0, MissingTomllib())
script, output = sys.argv[1:]
sys.argv = [script, '--skill-only', '--output', output]
runpy.run_path(script, run_name='__main__')
"""
    result = subprocess.run([sys.executable,'-B','-c',code,str(script),str(tmp_path / 'dist')],
                             capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
    with zipfile.ZipFile(tmp_path / 'dist/yue2-music.zip') as archive:
        assert archive.testzip() is None
        assert 'yue2-music/SKILL.md' in archive.namelist()
        assert 'yue2-music/instrumental/scripts/instrumental.py' in archive.namelist()
        for name in archive.namelist():
            assert archive.read(name) == (root / 'skills' / name).read_bytes()
