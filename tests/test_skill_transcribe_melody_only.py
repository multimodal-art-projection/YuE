"""CPU regressions for the portable helper's melody-only API contract."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import hashlib
import numpy as np
import pytest
import soundfile as sf
import torch
import transformers


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


ROOT = Path(__file__).resolve().parents[1]
ABC = 'X:1\nT:\nM:4/4\nL:1/32\nQ:1/4=88\nV: Vocal clef=treble name="Vocal Melody" snm="Vocal"\nV: Ins clef=treble name="Ins Melody" snm="Inst."\nK:C\nV: Vocal\nC8D8E8G8|\nV: Ins\nZ|\n'

@pytest.mark.parametrize('relative', ['scripts/transcribe.py', 'instrumental/scripts/transcribe_cover.py'])
@pytest.mark.parametrize('saved', ['changed', 'empty', 'same', 'crlf'])
def test_transcriber_checks_the_actual_saved_abc(tmp_path, monkeypatch, relative, saved):
    path = ROOT/'skills/yue2-music'/relative
    previous = {name: sys.modules.pop(name, None) for name in ('abc_tools', 'common')}
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location('transcriber_artifact_'+saved, path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        assert Path(module.parse_abc.__code__.co_filename) == path.parent/'abc_tools.py'
        assert Path(module.report.__code__.co_filename) == path.parent/'abc_tools.py'
        assert Path(module.sha256.__code__.co_filename) == path.parent/'common.py'
    finally:
        sys.path.pop(0)
        for name, original in previous.items():
            sys.modules.pop(name, None)
            if original is not None:
                sys.modules[name] = original
    audio = tmp_path/'source.wav'
    sf.write(audio, np.linspace(-.1,.1,256,dtype=np.float32), 48000, subtype='FLOAT')
    snapshot = tmp_path/'adapter'; snapshot.mkdir()
    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.gain = torch.nn.Parameter(torch.ones(())); self.config=SimpleNamespace(to_dict=lambda:{'fixture':'CPU adapter boundary'})
        def transcribe(self, source, output_dir, **kwargs):
            samples, rate = sf.read(source, dtype='float32')
            assert rate == 48000 and self(torch.from_numpy(samples)).numel() == 256
            value = ABC.replace('C8','D8') if saved=='changed' else '' if saved=='empty' else ABC.replace('\n','\r\n') if saved=='crlf' else ABC
            Path(output_dir,'score.abc').write_bytes(value.encode())
            return {'abc':ABC, 'warnings':[]}
        def forward(self, x): return x*self.gain
    monkeypatch.setattr(transformers.AutoModel,'from_pretrained',lambda *a,**k:Adapter())
    args = SimpleNamespace(audio=audio,output=tmp_path/'output',task='full',model=str(snapshot),revision=None,offline=True,base_model=None,preset='default',max_seconds=None,device='cpu',dtype='fp32',threads=1)
    if saved in ('changed','empty'):
        with pytest.raises(ValueError,match='Saved ABC'):
            module.run(args)
        assert not (args.output/'transcription_manifest.json').exists()
        assert json.loads((args.output/'failure.json').read_text())['status']=='failed'
    else:
        module.run(args)
        check=json.loads((args.output/'abc_check.json').read_text())
        assert check['score']['sha256']==hashlib.sha256((args.output/'score.abc').read_bytes()).hexdigest()
        assert check['status']=='passed'
        assert json.loads((args.output/'transcription_manifest.json').read_text())['status']=='complete'
