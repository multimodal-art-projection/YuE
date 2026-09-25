"""Batch request errors must not prevent the remaining songs from running."""
import json
from types import SimpleNamespace

import pytest

from yue2 import cli
from yue2.protocol import SongRequest


@pytest.mark.parametrize("invalid,error", [
    ({"abc_path": "missing.abc"}, "FileNotFoundError"),
    ({"unsupported": True}, "ValueError"),
])
def test_batch_continues_after_request_preflight_failure(invalid, error, monkeypatch, tmp_path):
    class FakePipe:
        weights = {}
        closed = False

        def __init__(self):
            self.generated = []

        def _request(self, **kwargs):
            return SongRequest(**kwargs)

        def effective_config(self, *args):
            return {}

        def __call__(self, **kwargs):
            self.generated.append(kwargs)
            return SimpleNamespace(save_artifacts=lambda directory: {"identity": "saved"})

        def close(self):
            self.closed = True

    pipe = FakePipe()
    monkeypatch.setattr(cli, "get_pipe", lambda args: pipe)
    source = tmp_path / "requests.jsonl"
    score = 'X:1\r\nK:C\r\nCDEF|\r\n'
    (tmp_path / "score.abc").write_bytes(score.encode("utf-8"))
    rows = [
        {"id": "invalid", "style": "piano", "lyrics": "First song", **invalid},
        {"id": "valid", "style": "jazz", "lyrics": "Second song", "abc_path": "score.abc"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "output"

    assert cli.main(["batch", "--input", str(source), "--output", str(output),
                     "--cot", "melody", "--quiet"]) == 1

    assert pipe.closed
    assert pipe.generated == [{"id": "valid", "style": "jazz", "lyrics": "Second song",
                               "abc": score, "cot": "melody"}]
    failure = json.loads((output / "invalid/failure.json").read_text())
    assert failure["type"] == error
    assert failure["status"] == "failed"
    receipt = json.loads((output / "batch.json").read_text())
    assert receipt["complete"] is False
    assert receipt["expected"] == 2
    assert receipt["failed"] == 1
    assert receipt["results"] == [failure, {"id": "valid", "status": "complete", "identity": "saved"}]
