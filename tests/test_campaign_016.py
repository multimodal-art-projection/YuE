import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import pytest
from yue2.pipeline import SymbolicPlan, SemanticResult, SongResult
from yue2.protocol import SongRequest

@pytest.mark.parametrize('optional', ['absent', 'listed', 'unlisted'])
def test_instrumental_share_only_exports_receipted_audio(tmp_path, optional):
    root = Path(__file__).resolve().parents[1]
    skill = root / 'skills/yue2-music/instrumental'
    script = skill / 'scripts/instrumental.py'
    prepared = tmp_path / 'prepared'
    env = dict(os.environ, PYTHONPATH=str(root / 'src'), PYTHONDONTWRITEBYTECODE='1')
    command = [sys.executable, '-B', str(script)]
    result = subprocess.run(command + ['prepare', '--abc', str(skill / 'assets/example-ins.abc'),
                                     '--output', str(prepared)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    request = SongRequest(**json.loads((prepared / 'request.json').read_text()))
    plan = SymbolicPlan(request, request.abc, [65], [1, 65])
    semantic = SemanticResult(plan, [1, 2], {}, False)
    audio = np.stack([np.sin(np.arange(4800) * 0.03), np.cos(np.arange(4800) * 0.03)], axis=-1) * 0.1
    song = SongResult(audio, 48000, semantic, np.zeros((1, 2, 2), dtype=np.float32), {}, {}, {}, 'fixture')
    source = tmp_path / 'generated'
    native = song.save_artifacts(source / 'native')
    for name in ('score.abc', 'prompt.txt', 'repair-report.json'):
        shutil.copyfile(prepared / name, source / name)
    summary = dict(status='complete', audio_seconds=native['audio_seconds'], sample_rate=48000,
                   decoder='native-fixture', seed=request.seed, cot=request.cot, package_version='fixture',
                   truncated=native['truncated'], audio_adherence_checked=False, absence_of_vocals_checked=False)
    (source / 'summary.json').write_text(json.dumps(summary))
    # Opaque optional delivery bytes exercise receipt coverage, not MP3 decoding.
    payload = b'optional-audio-receipt-fixture'
    if optional == 'listed': (source / 'listening.mp3').write_bytes(payload)
    files = {}
    for path in source.rglob('*'):
        if path.is_file():
            data = path.read_bytes()
            files[path.relative_to(source).as_posix()] = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    (source / 'delivery-manifest.json').write_text(json.dumps({'files': files}))
    if optional == 'unlisted': (source / 'listening.mp3').write_bytes(payload)
    output = tmp_path / 'listen'
    result = subprocess.run(command + ['share', str(source), '--output', str(output)],
                            env=env, capture_output=True, text=True)
    if optional == 'unlisted':
        assert result.returncode != 0
        assert 'listening.mp3' in result.stderr and 'manifest' in result.stderr
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert (output / 'audio.flac').read_bytes() == (source / 'native/audio.flac').read_bytes()
        if optional == 'listed':
            assert (output / 'listening.mp3').read_bytes() == payload
            assert 'src="listening.mp3"' in (output / 'index.html').read_text()
        else:
            assert not (output / 'listening.mp3').exists()
            assert 'src="audio.flac"' in (output / 'index.html').read_text()
