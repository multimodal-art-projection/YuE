"""Request preflight never loads the generation pipeline."""
import json
from pathlib import Path

import pytest

from yue2 import cli
from yue2.protocol import INSTRUCTIONS, SongRequest


ROOT = Path(__file__).resolve().parents[1]


def test_validate_is_a_public_command():
    args = cli.parser().parse_args(["validate", "--request", "examples/song.json"])
    assert args.command == "validate"
    assert args.request == "examples/song.json"


def test_validate_accepts_the_original_example(tmp_path, capsys):
    report_path = tmp_path / "report.json"
    assert cli.main(["validate", "--request", str(ROOT / "examples/song.json"),
                     "--output", str(report_path)]) == 0
    printed = json.loads(capsys.readouterr().out)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert printed == report
    assert report["ok"] is True
    assert report["checked"] == 1
    assert report["failed"] == 0
    row = report["results"][0]
    assert row["ok"] is True
    assert row["id"] == "city_lights"
    assert row["request"]["cot"] == "full"
    assert row["warnings"] == []


def test_validate_unknown_field_is_a_row_failure(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("validate must not load a pipeline")

    monkeypatch.setattr(cli, "get_pipe", forbidden)
    request = tmp_path / "bad.json"
    request.write_text(json.dumps({
        "id": "bad", "style": "piano", "lyrics": "[Verse]\nhello", "reference_audio": "x.wav",
    }), encoding="utf-8")
    assert cli.main(["validate", "--request", str(request)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert report["failed"] == 1
    assert report["results"][0]["type"] == "ValueError"
    assert "Unknown request fields" in report["results"][0]["error"]


def test_validate_batch_reports_each_row(tmp_path, capsys):
    score = tmp_path / "score.abc"
    score.write_bytes('X:1\r\nK:C\r\nCDEF|\r\n'.encode("utf-8"))
    source = tmp_path / "batch.jsonl"
    rows = [
        {"id": "ok_song", "style": "jazz", "lyrics": "[Verse]\nline"},
        {"id": "bad_song", "style": "jazz", "lyrics": "[Verse]\nline", "unknown": True},
        {"id": "with_score", "style": "jazz", "lyrics": "[Verse]\nline",
         "cot": "melody", "abc_path": "score.abc"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    assert cli.main(["validate", "--input", str(source)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["checked"] == 3
    assert report["failed"] == 1
    assert [row["ok"] for row in report["results"]] == [True, False, True]
    assert report["results"][2]["request"]["abc"] == "X:1\r\nK:C\r\nCDEF|\r\n"


def test_validate_duplicate_batch_ids_fail_closed(tmp_path, capsys):
    source = tmp_path / "dup.jsonl"
    row = {"id": "same", "style": "piano", "lyrics": "[Verse]\nline"}
    source.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    assert cli.main(["validate", "--input", str(source)]) == 2
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert report["checked"] == 0
    assert "unique id" in report["error"]


def test_validate_style_and_tags_must_agree(tmp_path, capsys):
    request = tmp_path / "tags.json"
    request.write_text(json.dumps({
        "id": "alias", "style": "piano", "tags": "guitar", "lyrics": "[Verse]\nline",
    }), encoding="utf-8")
    assert cli.main(["validate", "--request", str(request)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert "aliases" in report["results"][0]["error"]


def test_validate_accepts_tags_as_style(tmp_path, capsys):
    request = tmp_path / "tags-only.json"
    request.write_text(json.dumps({
        "id": "from_tags", "tags": "piano", "lyrics": "[Verse]\nline",
    }), encoding="utf-8")
    assert cli.main(["validate", "--request", str(request)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["results"][0]["request"]["style"] == "piano"


def test_validate_external_score_rejects_off_mode(tmp_path, capsys):
    abc = tmp_path / "score.abc"
    abc.write_text("X:1\nK:C\nCDEF|\n", encoding="utf-8")
    assert cli.main(["validate", "--style", "piano", "--lyrics", "[Verse]\nline",
                     "--abc-file", str(abc), "--cot", "off", "--id", "off_score"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["results"][0]["type"] == "ValueError"


def test_validate_cli_overrides_replace_request_fields(tmp_path, capsys):
    request = tmp_path / "base.json"
    request.write_text(json.dumps({
        "id": "old", "style": "piano", "lyrics": "[Verse]\nline", "cot": "full",
    }), encoding="utf-8")
    assert cli.main(["validate", "--request", str(request), "--id", "new",
                     "--cot", "melody"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["results"][0]["request"]["id"] == "new"
    assert report["results"][0]["request"]["cot"] == "melody"


def test_validate_historical_prompt_must_match(tmp_path, capsys):
    request = SongRequest("piano", "[Verse]\nline")
    path = tmp_path / "prompt.json"
    path.write_text(json.dumps({
        "id": "hist", "style": "piano", "lyrics": "[Verse]\nline", "prompt": request.text(),
    }), encoding="utf-8")
    assert cli.main(["validate", "--request", str(path)]) == 0
    capsys.readouterr()
    path.write_text(json.dumps({
        "id": "hist", "style": "piano", "lyrics": "[Verse]\nline",
        "prompt": INSTRUCTIONS["full"] + "\n[Tags]\nNOPE\n[Lyrics]\n[Verse]\nline\n",
    }), encoding="utf-8")
    assert cli.main(["validate", "--request", str(path)]) == 1
    assert "Historical literal prompt" in json.loads(capsys.readouterr().out)["results"][0]["error"]


def test_validate_invalid_sampling_is_reported(tmp_path, capsys):
    path = tmp_path / "sampling.json"
    path.write_text(json.dumps({
        "id": "hot", "style": "piano", "lyrics": "[Verse]\nline",
        "abc_sampling": {"temperature": -1},
    }), encoding="utf-8")
    assert cli.main(["validate", "--request", str(path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["results"][0]["ok"] is False


def test_validate_partial_sampling_keeps_defaults(tmp_path, capsys):
    path = tmp_path / "sampling.json"
    path.write_text(json.dumps({
        "id": "cool", "style": "piano", "lyrics": "[Verse]\nline",
        "abc_sampling": {"temperature": 0.4},
    }), encoding="utf-8")
    assert cli.main(["validate", "--request", str(path)]) == 0
    sampling = json.loads(capsys.readouterr().out)["results"][0]["sampling"]["abc_sampling"]
    assert sampling["temperature"] == 0.4
    assert sampling["max_tokens"] == 4096


def test_validate_flags_without_section_tags_warn(capsys):
    assert cli.main(["validate", "--style", "piano", "--lyrics", "no tags here",
                     "--id", "plain"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert "lyrics have no [Section] tags" in report["results"][0]["warnings"]


def test_validate_strict_promotes_warnings(capsys):
    assert cli.main(["validate", "--style", "piano", "--lyrics", "no tags here",
                     "--id", "plain", "--strict"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert report["results"][0]["type"] == "StrictWarning"


def test_validate_missing_abc_path(tmp_path, capsys):
    path = tmp_path / "missing.json"
    path.write_text(json.dumps({
        "id": "gone", "style": "piano", "lyrics": "[Verse]\nline", "abc_path": "nope.abc",
    }), encoding="utf-8")
    assert cli.main(["validate", "--request", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["results"][0]["type"] == "FileNotFoundError"


def test_validate_rejects_bad_id(capsys):
    assert cli.main(["validate", "--style", "piano", "--lyrics", "[Verse]\nline",
                     "--id", "../escape"]) == 1
    assert "filename-safe" in json.loads(capsys.readouterr().out)["results"][0]["error"]


def test_validate_empty_batch_is_a_command_error(tmp_path, capsys):
    source = tmp_path / "empty.jsonl"
    source.write_text("\n\n", encoding="utf-8")
    assert cli.main(["validate", "--input", str(source)]) == 2
    assert "no request objects" in json.loads(capsys.readouterr().out)["error"]


def test_validate_request_and_input_together_fail(tmp_path, capsys):
    request = tmp_path / "one.json"
    request.write_text(json.dumps({"id": "a", "style": "p", "lyrics": "[Verse]\nx"}), encoding="utf-8")
    source = tmp_path / "two.jsonl"
    source.write_text(request.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert cli.main(["validate", "--request", str(request), "--input", str(source)]) == 2
    assert "not both" in json.loads(capsys.readouterr().out)["error"]
