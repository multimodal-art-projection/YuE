import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import soundfile as sf
import torch
import transformers

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
