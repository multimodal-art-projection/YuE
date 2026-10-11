import json
from pathlib import Path
import pytest
from yue2 import cli
from yue2.protocol import SongRequest
from yue2.storage import collect_hashes, identity, write_json

class LoadedPipeline:
    weights = {}
    def __init__(self): self.closed = 0
    def _request(self, **kwargs): return SongRequest(**kwargs)
    def effective_config(self, *args): return {}
    def close(self): self.closed += 1

@pytest.mark.parametrize('mode', ['resume-corrupt','resume-complete','nonempty'])
def test_loaded_pipeline_is_closed_on_all_early_paths(tmp_path, monkeypatch, mode):
    pipe = LoadedPipeline()
    monkeypatch.setattr(cli,'get_pipe',lambda args: pipe)
    output = tmp_path/'output'; directory = output/'song'; directory.mkdir(parents=True)
    request = SongRequest('piano','words')
    if mode == 'nonempty':
        (directory/'existing.txt').write_text('prior result')
    else:
        for name in ['audio.flac','prefix.npy','semantic.npy','latent.npy','request.json','config.json']:
            (directory/name).write_bytes(b'storage-only fixture')
        write_json(directory/'result.json',{'status':'complete','truncated':False,
            'identity':identity({'request':request.to_dict(),'config':{},'weights':{}}),
            'artifacts':collect_hashes(directory) if mode == 'resume-complete' else {}})
    args = ['generate','--style','piano','--lyrics','words','--output',str(output),'--quiet']
    if mode != 'nonempty': args.append('--resume')
    if mode == 'resume-complete': assert cli.main(args) == 0
    else:
        with pytest.raises((ValueError,FileExistsError)): cli.main(args)
    assert pipe.closed == 1
