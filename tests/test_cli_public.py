"""Public installation resolves model IDs and exposes supported commands."""
import json
import pytest
from yue2 import cli, pipeline
from pathlib import Path
from yue2 import cli
from yue2.protocol import SongRequest
from yue2.storage import collect_hashes, identity, write_json
from types import SimpleNamespace
import sys
from yue2 import cli, storage


@pytest.mark.parametrize("vae,repo", [
    ("standard", "m-a-p/YuE2-Vae"),
    ("legacy", "m-a-p/YuE2-Vae-legacy"),
])
def test_cli_resolves_public_model_and_decoder(vae, repo, monkeypatch, tmp_path):
    monkeypatch.setenv("YUE2_KIT", str(tmp_path))
    args = cli.parser().parse_args(["generate", "--vae", vae])
    assert cli.model_paths(args) == ("m-a-p/YuE2-3B", repo)


def test_cli_accepts_explicit_model_and_decoder(monkeypatch, tmp_path):
    monkeypatch.setenv("YUE2_KIT", str(tmp_path))
    args = cli.parser().parse_args([
        "generate", "--model", "example/song-model", "--vae", "example/decoder",
    ])
    assert cli.model_paths(args) == ("example/song-model", "example/decoder")


@pytest.mark.parametrize("command", ["verify", "bench", "eval"])
def test_cli_rejects_commands_not_in_the_public_package(command):
    with pytest.raises(SystemExit) as exc:
        cli.parser().parse_args([command])
    assert exc.value.code == 2


def test_pipeline_defaults_and_explicit_revisions_reach_hub(monkeypatch, tmp_path):
    resolved = []

    def resolve(repo, **kwargs):
        resolved.append((repo, kwargs))
        return tmp_path / repo.rsplit("/", 1)[-1]

    class Pipeline(pipeline.YuE2Pipeline):
        def __init__(self, model, vae, **kwargs):
            self.model, self.vae, self.options = model, vae, kwargs
            self.load_timing = {}

    monkeypatch.setattr(pipeline, "resolve_model", resolve)
    result = Pipeline.from_pretrained(
        revision="a" * 40, vae_revision="b" * 40,
        token=False, cache_dir=str(tmp_path), progress=False,
    )
    assert [repo for repo, _ in resolved] == ["m-a-p/YuE2-3B", "m-a-p/YuE2-Vae"]
    assert [kwargs["revision"] for _, kwargs in resolved] == ["a" * 40, "b" * 40]
    assert all(kwargs["token"] is False for _, kwargs in resolved)
    assert result.options["progress"] is False


def test_saved_pipeline_uses_its_own_decoder(monkeypatch, tmp_path):
    (tmp_path / "pipeline.json").write_text(json.dumps({
        "model": "model", "vae": "decoder", "generation_config": {},
    }))
    resolved = []

    def resolve(path, **kwargs):
        resolved.append(path)
        return path

    class Pipeline(pipeline.YuE2Pipeline):
        def __init__(self, *args, **kwargs):
            self.load_timing = {}

    monkeypatch.setattr(pipeline, "resolve_model", resolve)
    Pipeline.from_pretrained(tmp_path, progress=False, local_files_only=True)
    assert resolved == [tmp_path / "model", tmp_path / "decoder"]


@pytest.mark.parametrize('identifier', ['../escaped','/absolute','nested/song','.', '..', '', 17, [], True])
def test_invalid_id_never_loads_a_pipeline_or_creates_outputs(tmp_path, monkeypatch, identifier):
    calls = []
    def load(args):
        calls.append(args)
        raise RuntimeError('model loader must not be entered')
    monkeypatch.setattr(cli, 'get_pipe', load)
    source = tmp_path / 'input.jsonl'
    source.write_text(json.dumps({'id':identifier, 'style':'piano','lyrics':'words'})+'\n')
    output = tmp_path / 'output'
    with pytest.raises(ValueError): cli.main(['batch','--input',str(source),'--output',str(output),'--quiet'])
    assert calls == [] and not output.exists()

def test_duplicate_ids_are_still_rejected_before_loading(tmp_path, monkeypatch):
    source = tmp_path/'input.jsonl'
    source.write_text(''.join(json.dumps({'id':'same','style':'piano','lyrics':'words'})+'\n' for _ in range(2)))
    def load(args): raise AssertionError('model loader must not be entered')
    monkeypatch.setattr(cli,'get_pipe',load)
    with pytest.raises(ValueError, match='unique'): cli.main(['batch','--input',str(source)])


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


def test_doctor_resolves_each_requested_revision_without_network(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv('YUE2_KIT',str(tmp_path))
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(xpu=SimpleNamespace(is_available=lambda:False),version=SimpleNamespace(hip=None),cuda=SimpleNamespace(device_count=lambda:0),backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda:False))))
    monkeypatch.setattr(cli.importlib.metadata,'version',lambda package:'synthetic-fixture-version')
    resolved = []
    def resolve(model, **kwargs):
        resolved.append((model, kwargs))
        return tmp_path
    monkeypatch.setattr(storage,'resolve_model',resolve)
    monkeypatch.setattr(storage,'model_identity',lambda path:{'files':{}})
    assert cli.main(['doctor','--verify-hashes','--offline','--revision','model-commit','--vae-revision','vae-commit']) == 0
    report = json.loads(capsys.readouterr().out)
    assert resolved == [('m-a-p/YuE2-3B',{'revision':'model-commit','local_files_only':True}),('m-a-p/YuE2-Vae',{'revision':'vae-commit','local_files_only':True})]
    assert report['validated'] is False
