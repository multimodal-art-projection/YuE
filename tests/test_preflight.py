import json

import pytest

from yue2 import cli
from yue2.protocol import GenerationConfig, preflight


@pytest.mark.parametrize("identifier", ["../escape", "/absolute", "a/b", "a\\b", "", None, 4, []])
def test_single_output_id_is_checked_before_loading(tmp_path, monkeypatch, identifier):
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"id": identifier, "style": "piano", "lyrics": ""}))
    monkeypatch.setattr(cli, "get_pipe", lambda _: pytest.fail("model loaded"))
    output = tmp_path / "output"
    with pytest.raises((TypeError, ValueError)):
        cli.main(["generate", "--request", str(request), "--output", str(output)])
    assert not output.exists()


def test_shared_preflight_accepts_instrumentals_and_custom_sampling_defaults(tmp_path, monkeypatch, capsys):
    request = {"style": "piano", "lyrics": "", "semantic_sampling": {"max_tokens": 2}}
    config = GenerationConfig.from_dict({"semantic": {"min_tokens": 0}})
    assert preflight(request, config).lyrics == ""
    with pytest.raises(ValueError):
        preflight(request)
    source, settings = tmp_path / "song.json", tmp_path / "config.json"
    source.write_text(json.dumps(request))
    settings.write_text(json.dumps(config.to_dict()))
    monkeypatch.setattr(cli, "get_pipe", lambda _: pytest.fail("model loaded"))
    assert cli.main(["validate", "--request", str(source), "--config", str(settings)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    assert cli.main(["validate", "--request", str(source)]) == 1


def test_batch_with_only_invalid_requests_never_loads_models(tmp_path, monkeypatch):
    source = tmp_path / "songs.jsonl"
    source.write_text(json.dumps({"id": "bad", "style": "piano", "lyrics": "", "seed": -1}) + "\n")
    monkeypatch.setattr(cli, "get_pipe", lambda _: pytest.fail("model loaded"))
    assert cli.main(["batch", "--input", str(source), "--output", str(tmp_path / "out"), "--quiet"]) == 1


def test_aliases_have_one_validation_rule():
    assert preflight({"tags": "piano", "lyrics": ""}).style == "piano"
    with pytest.raises(ValueError, match="aliases"):
        preflight({"style": "piano", "tags": "jazz", "lyrics": ""})
