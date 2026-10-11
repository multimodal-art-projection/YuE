from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import pytest

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
