import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import soundfile as sf
from yue2.pipeline import SymbolicPlan, SemanticResult, SongResult
from yue2.protocol import SongRequest
from yue2.storage import verify_result

@pytest.fixture
def saved(tmp_path):
    request=SongRequest(style='piano',lyrics='[Verse]\nHello',cot='off')
    plan=SymbolicPlan(request,None,[],[1,2],{'seconds':0.1},False)
    semantic=SemanticResult(plan,[3,4],{'seconds':0.2},True)
    audio=np.stack([np.sin(np.arange(4800)*0.03),np.cos(np.arange(4800)*0.03)],axis=-1)*0.1
    song=SongResult(audio,48000,semantic,np.zeros((1,64,2),dtype=np.float32),{}, {},{},'fixture-id')
    directory=tmp_path/'native';song.save_artifacts(directory)
    return directory

def inspect(directory,tmp_path):
    boundary=tmp_path/'no-models';boundary.mkdir(exist_ok=True)
    (boundary/'sitecustomize.py').write_text("import sys\nclass Boundary:\n    def find_spec(self, fullname, path=None, target=None):\n        if fullname.split('.')[0] in {'torch','transformers','tiktoken'}: raise RuntimeError('inspection imported model runtime')\nsys.meta_path.insert(0,Boundary())\n")
    env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(boundary), str(Path(__file__).resolve().parents[1] / 'src'))))
    return subprocess.run([sys.executable,'-m','yue2.cli','inspect',str(directory)],env=env,capture_output=True,text=True)

def test_inspect_verifies_saved_audio_without_model_runtime(saved,tmp_path):
    result=inspect(saved,tmp_path)
    assert result.returncode==0,result.stderr
    record=json.loads(result.stdout)
    audio,rate=sf.read(saved/'audio.flac',dtype='float32')
    assert record['artifacts_verified'] is True
    assert record['identity']=='fixture-id'
    assert record['sample_rate']==rate==48000
    assert record['audio_seconds']==len(audio)/rate
    assert record['truncated']=={'abc':False,'semantic':True}
    assert record['needs_review'] is True
    assert record['quality_checked'] is False

def test_inspect_rejects_corrupt_audio(saved,tmp_path):
    with (saved/'audio.flac').open('ab') as stream: stream.write(b'changed')
    result=inspect(saved,tmp_path)
    assert result.returncode!=0 and 'corrupt result: audio.flac' in result.stderr
    assert result.stdout==''

def test_existing_native_result_verification_control(saved):
    record=verify_result(saved)
    assert record['status']=='complete' and record['truncated']['semantic'] is True
