"""Public installation resolves model IDs and exposes supported commands."""
import json
import pytest

from yue2 import cli, pipeline


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
