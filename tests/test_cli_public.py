"""Public installation resolves model IDs and exposes supported commands."""
import json
import os
import pytest

from yue2 import cli, pipeline
from yue2.protocol import GenerationConfig, SongRequest


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


def test_miopen_find_mode_is_opt_in_and_preserves_inherited_value(monkeypatch):
    monkeypatch.setenv("MIOPEN_FIND_MODE", "HYBRID")
    args = cli.parser().parse_args(["generate"])

    cli.apply_runtime_options(args)

    assert args.miopen_find_mode is None
    assert os.environ["MIOPEN_FIND_MODE"] == "HYBRID"


def test_miopen_fast_is_applied_before_command_dispatch(monkeypatch):
    monkeypatch.delenv("MIOPEN_FIND_MODE", raising=False)
    observed = {}

    def generate(args):
        observed["mode"] = os.environ.get("MIOPEN_FIND_MODE")
        return 0

    monkeypatch.setattr(cli, "generate", generate)
    assert cli.main(["generate", "--miopen-find-mode", "FAST"]) == 0
    assert observed == {"mode": "FAST"}


def test_cli_rejects_unsupported_miopen_find_modes():
    with pytest.raises(SystemExit) as exc:
        cli.parser().parse_args(["generate", "--miopen-find-mode", "NORMAL"])
    assert exc.value.code == 2


def test_effective_config_records_miopen_find_mode(monkeypatch, tmp_path):
    monkeypatch.setenv("MIOPEN_FIND_MODE", "FAST")
    (tmp_path / "config.json").write_text(json.dumps({"release_variant": "test"}))
    pipe = object.__new__(pipeline.YuE2Pipeline)
    pipe.generation_config = GenerationConfig()
    pipe.backend, pipe.quantization = "torch", "none"
    pipe.vae_core_frames, pipe.memory_budget_gib = 1024, 24.0
    pipe.__dict__["device"] = "cpu"
    pipe.offload_ar = False
    pipe.runtime_sha256, pipe.vae_dir = "runtime", tmp_path

    config = pipe.effective_config(SongRequest(style="style", lyrics="lyrics"))

    assert config["runtime_environment"] == {"miopen_find_mode": "FAST"}


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
