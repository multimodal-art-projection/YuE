"""The skill must retain explicit score edits and reject ambiguous requests."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "skills/yue2-music/scripts"
with patch.object(sys, "path", [str(SCRIPTS), *sys.path]):
    SPEC = importlib.util.spec_from_file_location("skill_requests", SCRIPTS / "run_yue2.py")
    MODULE = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(MODULE)

ABC = '''X:1
T:
M:4/4
L:1/32
Q:1/4=88
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:C
% verse
V: Vocal
"C"C8D8E8G8|
V: Ins
Z|
'''


class RequestChecks(unittest.TestCase):
    def prepare(self, directory, request, *, abc_file=None, cot=None, action="generate"):
        path = directory / "request.json"
        path.write_text(json.dumps({"style": "Piano pop", "lyrics": "A new day", **request}))
        return SimpleNamespace(request=path, abc_file=abc_file, cot=cot, action=action)

    def test_explicit_edited_file_survives_null_inline_abc(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            score = root / "edited.abc"
            score.write_text(ABC)
            args = self.prepare(root, {"abc": None}, abc_file=score)
            self.assertEqual(MODULE.request_data(args)["abc"], ABC)

    def test_score_path_resolves_relative_to_request(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "edited.abc").write_text(ABC)
            args = self.prepare(root, {"abc_path": "edited.abc"})
            result = MODULE.request_data(args)
            self.assertEqual(result["abc"], ABC)
            self.assertNotIn("abc_path", result)

    def test_multiple_score_sources_fail_before_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.prepare(root, {"abc": ABC}, abc_file=root / "other.abc")
            with self.assertRaisesRegex(ValueError, "only one"):
                MODULE.request_data(args)

    def test_cover_rejects_chords_until_explicitly_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.prepare(root, {"abc": ABC}, cot="melody")
            with self.assertRaisesRegex(ValueError, "still contains chords"):
                MODULE.request_data(args)
            args = self.prepare(root, {"abc": ABC.replace('"C"', '')}, cot="melody")
            self.assertNotIn('"C"', MODULE.request_data(args)["abc"])

    def test_score_input_cannot_be_silently_ignored_by_off_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.prepare(Path(directory), {"abc": ABC}, cot="off")
            with self.assertRaisesRegex(ValueError, "off cannot accept ABC"):
                MODULE.request_data(args)

    def test_unsupported_audio_reference_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.prepare(Path(directory), {"reference_audio": "source.wav"})
            with self.assertRaisesRegex(ValueError, "Unsupported request fields"):
                MODULE.request_data(args)


if __name__ == "__main__":
    unittest.main(verbosity=2)
