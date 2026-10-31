"""Lyric/score section alignment is a skill helper and does not load models."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills/yue2-music/scripts/abc_tools.py"


def run_align(*args):
    return subprocess.run([sys.executable, "-B", str(SCRIPT), "align-lyrics", *args],
                          capture_output=True, text=True)


def test_align_lyrics_cli_accepts_city_lights():
    result = run_align("--request", str(ROOT / "examples/song.json"),
                       "--score", str(ROOT / "examples/score.abc"),
                       "--require-line-per-bar", "--require-onset-syllables")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["match"] is True
    assert report["differences"] == []
    assert [pair["lyric_name"] for pair in report["pairs"]] == ["verse", "chorus"]


def test_align_lyrics_cli_writes_report(tmp_path):
    output = tmp_path / "phrasing.json"
    result = run_align("--request", str(ROOT / "examples/song.json"),
                       "--score", str(ROOT / "examples/melody.abc"),
                       "--output", str(output))
    assert result.returncode == 0, result.stderr
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["match"] is True
    assert output.exists()


def test_align_lyrics_cli_detects_missing_chorus(tmp_path):
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("[Verse]\nNeon fades along the lane\n", encoding="utf-8")
    result = run_align("--lyrics", str(lyrics), "--score", str(ROOT / "examples/score.abc"))
    assert result.returncode == 1
    report = json.loads(result.stdout)
    assert report["match"] is False
    assert any("section count" in item for item in report["differences"])


def test_align_lyrics_cli_rejects_both_inputs(tmp_path):
    lyrics = tmp_path / "lyrics.txt"
    lyrics.write_text("[Verse]\nline\n", encoding="utf-8")
    result = run_align("--lyrics", str(lyrics), "--request", str(ROOT / "examples/song.json"),
                       "--score", str(ROOT / "examples/score.abc"))
    assert result.returncode == 2
    assert "exactly one" in result.stderr


def test_align_lyrics_refuses_to_overwrite_report(tmp_path):
    output = tmp_path / "phrasing.json"
    output.write_text("{}\n", encoding="utf-8")
    result = run_align("--request", str(ROOT / "examples/song.json"),
                       "--score", str(ROOT / "examples/score.abc"),
                       "--output", str(output))
    assert result.returncode == 2
    assert "already exists" in result.stderr
