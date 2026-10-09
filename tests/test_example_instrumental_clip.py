"""The short-clip example must transfer the Vocal melody, cap tokens to the requested length, and trim."""
import dataclasses
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("example_instrumental_clip", ROOT / "examples/instrumental_clip.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@dataclasses.dataclass(frozen=True)
class FakeSampling:
    max_tokens: int = 9000


def test_style_gets_instrumental_conditions():
    style = MODULE.instrumental_style("warm solo piano.")
    assert style.startswith("Instrumental, warm solo piano")
    assert all(x in style for x in ("no vocals", "no singing", "no choir", "no spoken words"))
    assert MODULE.instrumental_style("Instrumental piano, no vocals").count("no vocals") == 1


def test_clip_moves_vocal_to_instrumental_and_caps_tokens(tmp_path, monkeypatch):
    score = (ROOT / "examples/score.abc").read_bytes().decode("utf-8").replace("\r\n", "\n")
    pipe = MagicMock()
    pipe.generation_config = SimpleNamespace(semantic=FakeSampling())
    pipe.plan.side_effect = [SimpleNamespace(truncated=False, abc=score), "final-plan"]
    pipe.decode.return_value = np.zeros((48000 * 5, 2), dtype=np.float32)
    loader = MagicMock()
    loader.from_pretrained.return_value.__enter__.return_value = pipe
    written = {}
    monkeypatch.setitem(sys.modules, "yue2", SimpleNamespace(YuE2Pipeline=loader))
    monkeypatch.setitem(sys.modules, "yue2.protocol", SimpleNamespace(SongRequest=lambda **kw: SimpleNamespace(**kw)))
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace(write=lambda path, audio, rate: written.update(
        path=path, frames=len(audio), rate=rate)))
    output = tmp_path / "clip.flac"
    saved = tmp_path / "clip.abc"
    monkeypatch.setattr(sys, "argv", [str(ROOT / "examples/instrumental_clip.py"), "--style", "warm solo piano",
        "--seconds", "3", "--seed", "7", "--output", str(output), "--save-score", str(saved)])

    assert MODULE.main() == 0
    request = pipe.plan.call_args_list[1].kwargs
    assert request["seed"] == 7 and request["cot"] in ("full", "melody")
    assert "no vocals" in request["style"]
    from abc_tools import parse_abc
    before, after = parse_abc(score), parse_abc(request["abc"])
    assert before.voices["Vocal"].notes and not after.voices["Vocal"].notes
    assert len(after.voices["Ins"].notes) > len(before.voices["Ins"].notes)
    assert saved.read_text(encoding="utf-8") == request["abc"]
    pipe.generate_semantic.assert_called_once()
    assert pipe.generate_semantic.call_args.kwargs["sampling"].max_tokens == 3 * 25 + 250
    assert written == {"path": output, "frames": 48000 * 3, "rate": 48000}
