"""The documented example must pass external scores to the pipeline unchanged."""
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("example_generate", ROOT / "examples/generate.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("newline", ["\n", "\r\n", "mixed"], ids=["lf", "crlf", "mixed"])
@pytest.mark.parametrize("device", [None, "mps"])
def test_example_preserves_external_score(newline, device, tmp_path, monkeypatch):
    score = (ROOT / "examples/score.abc").read_bytes().decode("utf-8").replace("\r\n", "\n")
    score = score.replace("% verse", "% verse — mélodie")
    if newline == "mixed":
        score = score.replace("\n", "\r\n", 3)
    else:
        score = score.replace("\n", newline)
    abc_file = tmp_path / "edited.abc"
    abc_file.write_bytes(score.encode("utf-8"))
    request = {"style": "Piano pop", "lyrics": "A new day", "cot": "full"}
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps(request), encoding="utf-8")
    output = tmp_path / "output"

    song = SimpleNamespace(truncated={"abc": False, "semantic": False},
                           save_artifacts=MagicMock())
    loader = MagicMock()
    pipe = loader.from_pretrained.return_value.__enter__.return_value
    pipe.return_value = song
    monkeypatch.setitem(sys.modules, "yue2", SimpleNamespace(YuE2Pipeline=loader))
    argv = [str(ROOT / "examples/generate.py"), "--request", str(request_file),
            "--abc-file", str(abc_file), "--output", str(output)]
    if device:
        argv += ["--device", device]
    monkeypatch.setattr(sys, "argv", argv)

    assert MODULE.main() == 0
    assert loader.from_pretrained.call_args.kwargs["device"] == (device or "auto")
    pipe.assert_called_once_with(**request, abc=score)
    song.save_artifacts.assert_called_once_with(output)
    assert abc_file.read_bytes() == score.encode("utf-8")


def test_instrumental_clip_rejects_invalid_duration_before_loading(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    script = Path(__file__).resolve().parents[1] / 'examples/instrumental_clip.py'
    for seconds in ['nan', 'inf', '-1', '0', '0.00000001']:
        result = subprocess.run([sys.executable, str(script), '--prepared', str(tmp_path / 'absent'),
                                 '--seconds', seconds, '--output', str(tmp_path / 'output')],
                                capture_output=True, text=True)
        assert result.returncode == 2
        assert 'seconds must' in result.stderr
    assert not (tmp_path / 'output').exists()
