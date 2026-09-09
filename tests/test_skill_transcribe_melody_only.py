"""CPU regressions for the portable helper's melody-only API contract."""
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
    SPEC = importlib.util.spec_from_file_location("skill_transcribe", SCRIPTS / "transcribe.py")
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
C8D8E8G8|
V: Ins
Z|
'''


class Model:
    def __init__(self):
        self.config = SimpleNamespace(to_dict=lambda: {})
        self.calls = []

    def eval(self):
        return self

    def to(self, device):
        return self

    def transcribe(self, audio, output_dir=None, *, melody_only=False, **kwargs):
        self.calls.append(dict(melody_only=melody_only, **kwargs))
        Path(output_dir, "score.abc").write_text(ABC)
        return {"abc": ABC, "abc_error": None, "warnings": []}


class OldModel(Model):
    def transcribe(self, audio, output_dir=None, **kwargs):
        self.legacy_kwargs = dict(kwargs)
        return super().transcribe(audio, output_dir, **kwargs)


class TranscribeMelodyOnlyChecks(unittest.TestCase):
    def run_helper(self, root, model, task):
        source = root / "audio.wav"
        source.write_bytes(b"Synthetic CPU mock input; not an audio measurement")
        args = SimpleNamespace(
            audio=source, output=root / "output", task=task,
            model=str(root / "mock-model"), revision=None, base_model=None,
            offline=True, device="cpu", dtype="fp32", preset="default",
            max_seconds=None, threads=1,
        )
        modules = {
            "torch": SimpleNamespace(set_num_threads=lambda _: None),
            "transformers": SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=lambda *a, **k: model)),
        }
        with patch.dict(sys.modules, modules), patch.object(MODULE.importlib.metadata, "version", return_value="mock"):
            MODULE.run(args)
        return args.output

    def test_melody_tasks_explicitly_enable_export_flag(self):
        for task, prompt in (("melody-full", "melody_full"), ("melody-vocal", "melody_vocal")):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                model = Model()
                output = self.run_helper(Path(directory), model, task)
                self.assertIs(model.calls[0]["melody_only"], True)
                self.assertIn(prompt, model.calls[0]["prompts"])
                self.assertNotIn("chord_full", model.calls[0]["prompts"])
                self.assertIs(json.loads((output / "input.json").read_text())["melody_only"], True)
                self.assertEqual(json.loads((output / "abc_check.json").read_text())["status"], "passed")

    def test_full_task_remains_compatible_with_old_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            model = OldModel()
            output = self.run_helper(Path(directory), model, "full")
            self.assertIs(model.calls[0]["melody_only"], False)
            self.assertNotIn("melody_only", model.legacy_kwargs)
            self.assertIn("chord_full", model.calls[0]["prompts"])
            self.assertIs(json.loads((output / "input.json").read_text())["melody_only"], False)

    def test_old_wrapper_rejected_before_transcribing_melody(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = OldModel()
            with self.assertRaisesRegex(RuntimeError, "refresh the model"):
                self.run_helper(root, model, "melody-full")
            self.assertEqual(model.calls, [])
            failure = json.loads((root / "output/failure.json").read_text())
            self.assertEqual(failure["status"], "failed")
            self.assertIn("melody_only", failure["error"])

    def test_missing_abc_remains_failure_with_partial_error_result(self):
        class FailedModel(Model):
            def transcribe(self, audio, output_dir=None, *, melody_only=False, **kwargs):
                error = RuntimeError("Melody-only ABC unavailable: No decoded beats")
                error.result = {"abc": None, "abc_error": "No decoded beats", "events": []}
                raise error
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "No decoded beats") as stopped:
                self.run_helper(root, FailedModel(), "melody-full")
            self.assertEqual(stopped.exception.result["abc_error"], "No decoded beats")
            self.assertTrue((root / "output/failure.json").is_file())
            self.assertFalse((root / "output/transcription_manifest.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
