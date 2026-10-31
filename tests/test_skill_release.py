"""Check that the downloadable skill includes its fixtures and excludes local data."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
import zipfile
import sys
import pytest
import os


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("skill_release", ROOT / "tools/build_release.py")
BUILD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILD)


class SkillReleaseTests(unittest.TestCase):
    def test_reproducible_archive_contains_complete_instrumental_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            first = BUILD.build_skill(ROOT, Path(directory) / "first")
            second = BUILD.build_skill(ROOT, Path(directory) / "second")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with zipfile.ZipFile(first) as archive:
                names = set(archive.namelist())
                for relative in ("SKILL.md", "scripts/run_yue2.py", "instrumental/SKILL.md",
                                 "instrumental/scripts/instrumental.py", "instrumental/scripts/transcribe_cover.py",
                                 "instrumental/assets/example-ins.abc", "instrumental/assets/example-style.txt",
                                 "instrumental/requirements-inference.txt", "instrumental/bundle-manifest.json"):
                    self.assertIn("yue2-music/" + relative, names)
                for member in archive.infolist():
                    self.assertFalse(member.extra or member.comment)
                    self.assertFalse(member.filename.endswith((".mp3", ".flac", ".npy", ".log", ".pyc")))

    def fixture(self, root):
        skill = root / "skills/yue2-music"
        shutil.copytree(ROOT / "skills/yue2-music", skill)
        return skill

    def test_rejects_unlisted_private_work_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill = self.fixture(root)
            (skill / "instrumental/.private-run.log").write_text("Synthetic private work file")
            with self.assertRaises(subprocess.CalledProcessError):
                BUILD.build_skill(root, root / "dist")
            self.assertFalse((root / "dist/yue2-music.zip").exists())

    def test_rejects_changed_manifest_payload(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skill = self.fixture(root)
            (skill / "instrumental/assets/example-style.txt").write_text("Changed without updating manifest")
            with self.assertRaises(subprocess.CalledProcessError):
                BUILD.build_skill(root, root / "dist")
            self.assertFalse((root / "dist/yue2-music.zip").exists())


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize('variation', ['extra-asset','extra-script','missing-script','clean'])
def test_public_skill_builder_enforces_exact_top_level_file_boundary(tmp_path, variation):
    root = Path(__file__).resolve().parents[1]
    staged = tmp_path / 'source'
    skill = staged / 'skills/yue2-music'
    shutil.copytree(root / 'skills/yue2-music',skill)
    (staged / 'tools').mkdir()
    shutil.copyfile(root / 'tools/build_release.py',staged / 'tools/build_release.py')
    if variation.startswith('extra'):
        folder = 'assets' if variation == 'extra-asset' else 'scripts'
        (skill / folder / 'private-session.json').write_text('PRIVATE_TEST_ONLY_DO_NOT_EXPORT')
    elif variation == 'missing-script': (skill / 'scripts/transcribe.py').unlink()
    output = tmp_path / 'dist'
    result = subprocess.run([sys.executable,'-B',str(staged / 'tools/build_release.py'),
                             '--skill-only','--output',str(output)],capture_output=True,text=True)
    if variation != 'clean':
        assert result.returncode != 0
        assert not (output / 'yue2-music.zip').exists()
    else:
        assert result.returncode == 0, result.stderr
        with zipfile.ZipFile(output / 'yue2-music.zip') as archive:
            main = {name.removeprefix('yue2-music/') for name in archive.namelist()
                    if not name.startswith('yue2-music/instrumental/')}
            assert main == {'SKILL.md','LICENSE','agents/openai.yaml','assets/edit-brief.md','assets/prompt.json',
                'scripts/abc_tools.py','scripts/common.py','scripts/listen.py','scripts/run_yue2.py','scripts/transcribe.py',
                'references/abc-editing.md','references/editing-workflows.md','references/generation-and-covers.md',
                'references/listening-and-evaluation.md','references/models-and-setup.md'}
            assert archive.testzip() is None
            for name in archive.namelist():
                assert archive.read(name) == (skill / name.removeprefix('yue2-music/')).read_bytes()


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
