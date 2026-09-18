"""Shipped code must not depend on the platform's default text encoding.

Python resolves ``encoding=None`` through ``locale.getpreferredencoding``,
which is UTF-8 on Linux but cp1252 on a default Windows install. Requests,
plans and results are written with ``ensure_ascii=False``, so any non-ASCII
lyric or ABC annotation round-trips through those code paths as raw UTF-8.
"""

import ast
import json
from pathlib import Path

import pytest

from yue2 import SymbolicPlan
from yue2.protocol import SongRequest, token_prefixes
from yue2.storage import write_json

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = (ROOT / "src" / "yue2", ROOT / "skills" / "yue2-music" / "scripts")
TEXT_IO = {"read_text", "write_text"}

# A Mandarin request like the CLI's own default, with an ABC annotation that
# also leaves ASCII.
LYRICS = "[Verse]\n晚风轻轻吹过窗前\n你留下的笑还在昨天"
SCORE = 'X:1\nT:夜曲\nK:C\nw: Café où l\'été\n"C"C8D8|\n'


class Tokenizer:
    def encode(self, text):
        return [ord(x) for x in text]

    def decode(self, ids):
        return "".join(chr(x) for x in ids)


def implicit_text_io(path):
    """Yield (line, call) for pathlib text helpers that omit ``encoding``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in TEXT_IO:
            continue
        if not any(word.arg == "encoding" for word in node.keywords):
            yield node.lineno, node.func.attr


def test_shipped_text_io_declares_an_encoding():
    offenders = [
        f"{path.relative_to(ROOT).as_posix()}:{line} {call}()"
        for root in SHIPPED
        for path in sorted(root.rglob("*.py"))
        for line, call in implicit_text_io(path)
    ]
    assert offenders == [], "text I/O without an explicit encoding: " + ", ".join(offenders)


def test_saved_plan_round_trips_non_ascii(tmp_path):
    request = SongRequest("Mandarin, 温暖钢琴", LYRICS, abc=SCORE)
    ids = Tokenizer().encode(SCORE)
    plan = SymbolicPlan(request, SCORE, ids, token_prefixes(request, Tokenizer(), ids))
    plan.save(tmp_path)
    assert SymbolicPlan.load(tmp_path) == plan


def test_result_manifest_round_trips_non_ascii(tmp_path):
    write_json(tmp_path / "request.json", {"lyrics": LYRICS, "style": "温暖钢琴"})
    assert json.loads((tmp_path / "request.json").read_text(encoding="utf-8"))["lyrics"] == LYRICS


@pytest.mark.parametrize("text", [LYRICS, SCORE, "Grüße", "Corazón", "夜の窓辺"])
def test_write_json_round_trips_without_mojibake(tmp_path, text):
    path = tmp_path / "value.json"
    write_json(path, {"text": text})
    assert json.loads(path.read_text(encoding="utf-8"))["text"] == text
