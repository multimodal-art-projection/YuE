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
def test_example_preserves_external_score(newline, tmp_path, monkeypatch):
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
    monkeypatch.setattr(sys, "argv", [str(ROOT / "examples/generate.py"),
        "--request", str(request_file), "--abc-file", str(abc_file), "--output", str(output)])

    assert MODULE.main() == 0
    pipe.assert_called_once_with(**request, abc=score)
    song.save_artifacts.assert_called_once_with(output)
    assert abc_file.read_bytes() == score.encode("utf-8")
