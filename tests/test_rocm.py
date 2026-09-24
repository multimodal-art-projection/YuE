"""ROCm policy contracts without a GPU or model downloads."""
import json
from types import SimpleNamespace

import pytest
import torch

from yue2 import cli, pipeline, rocm, sampling
from yue2.protocol import GenerationConfig, SongRequest
from yue2.storage import identity


@pytest.fixture
def runtime(monkeypatch):
    devices = []

    def properties(device):
        devices.append(device)
        return SimpleNamespace(gcnArchName="gfx1151:sramecc-:xnack-")

    fake = SimpleNamespace(
        __version__=rocm.TESTED_TORCH,
        version=SimpleNamespace(hip="7.2.53211-158bd99533"),
        cuda=SimpleNamespace(get_device_properties=properties),
        backends=SimpleNamespace(cudnn=SimpleNamespace(enabled=True)),
    )
    monkeypatch.setattr(rocm, "torch", fake)
    monkeypatch.setattr(rocm, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.delenv(rocm.AOTRITON_ENV, raising=False)
    return fake, devices


def configure(profile="windows-gfx1151", device="cuda:1", **kwargs):
    rocm.configure_profile(profile, torch.device(device),
                           backend=kwargs.get("backend", "torch"),
                           quantization=kwargs.get("quantization", "none"))


@pytest.mark.parametrize("value", [None, "0", "1"])
def test_profile_is_explicit_and_respects_attention_opt_out(runtime, monkeypatch, value):
    if value is not None:
        monkeypatch.setenv(rocm.AOTRITON_ENV, value)
    with pytest.warns(RuntimeWarning, match="Experimental"):
        configure()
    assert rocm.os.environ[rocm.AOTRITON_ENV] == ("1" if value is None else value)
    assert runtime[1] == [torch.device("cuda:1")]


def test_default_does_not_probe_hardware_or_change_environment(runtime):
    configure(None)
    assert runtime[1] == [] and rocm.AOTRITON_ENV not in rocm.os.environ


@pytest.mark.parametrize("change", ["nvidia", "cpu", "mps", "linux", "version", "architecture", "vllm", "fp8", "unknown"])
def test_unsupported_profile_fails_before_enabling_experimental_kernels(runtime, monkeypatch, change):
    fake, _ = runtime
    options = {}
    if change == "nvidia":
        fake.version.hip = None
    elif change in {"cpu", "mps"}:
        options["device"] = change
    elif change == "linux":
        monkeypatch.setattr(rocm, "sys", SimpleNamespace(platform="linux"))
    elif change == "version":
        fake.__version__ = "2.10.0+rocm7.2"
    elif change == "architecture":
        fake.cuda.get_device_properties = lambda device: SimpleNamespace(gcnArchName="gfx1100")
    elif change == "vllm":
        options["backend"] = "vllm"
    elif change == "fp8":
        options["quantization"] = "fp8"
    elif change == "unknown":
        options["profile"] = "unknown"
    with pytest.raises(ValueError):
        configure(**options)
    assert rocm.AOTRITON_ENV not in rocm.os.environ


@pytest.mark.parametrize("previous", [True, False])
@pytest.mark.parametrize("failure", [None, RuntimeError, InterruptedError, KeyboardInterrupt])
def test_native_decoder_restores_flag_even_on_failure_or_cancellation(runtime, previous, failure):
    flags = runtime[0].backends.cudnn
    flags.enabled = previous
    try:
        with rocm.decoder_execution("windows-gfx1151"):
            assert flags.enabled is False
            if failure is not None:
                raise failure("test")
    except BaseException as exc:
        assert failure is not None and type(exc) is failure
    assert flags.enabled is previous


def test_default_decoder_and_nested_contexts_restore_correctly(runtime):
    flags = runtime[0].backends.cudnn
    with rocm.decoder_execution(None):
        assert flags.enabled is True
    with rocm.decoder_execution("windows-gfx1151"):
        with rocm.decoder_execution("windows-gfx1151"):
            assert flags.enabled is False
        assert flags.enabled is False
    assert flags.enabled is True


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("failure", [False, True])
def test_pipeline_wraps_full_and_tiled_decode_only(runtime, full, failure):
    flags = runtime[0].backends.cudnn
    calls = []

    class Decoder:
        def to(self, device):
            assert flags.enabled is True
            calls.append(str(device))
            return self

        def decode(self, z, **kwargs):
            assert flags.enabled is False
            assert z.dtype == torch.float32
            if failure:
                raise RuntimeError("decoder failed")
            return torch.zeros(1, 2, 16)

        decode_tiled = decode

    pipe = object.__new__(pipeline.YuE2Pipeline)
    pipe.rocm_profile, pipe.progress = "windows-gfx1151", False
    pipe.device, pipe._model, pipe._vae = torch.device("cpu"), None, Decoder()
    pipe.vae_core_frames = 1024
    if failure:
        with pytest.raises(RuntimeError, match="decoder failed"):
            pipe.decode(torch.zeros(2, 64), full=full)
    else:
        assert pipe.decode(torch.zeros(2, 64), full=full).shape == (16, 2)
    assert flags.enabled is True and calls == ["cpu", "cpu"]


@pytest.mark.parametrize("device,hip,fp8,reason", [
    ("cpu", None, False, "non_cuda_device"),
    ("cpu", "7.2", False, "non_cuda_device"),
    ("mps", None, False, "non_cuda_device"),
    ("cuda:0", "7.2", False, "rocm_not_graph_validated"),
    ("cuda:0", "7.2", True, "rocm_not_graph_validated"),
    ("cuda:0", None, True, "fp8_not_graph_validated"),
    ("cuda:0", None, False, None),
])
def test_graph_selection_keeps_nvidia_default_and_avoids_rocm_varlen(device, hip, fp8, reason, monkeypatch):
    monkeypatch.setattr(sampling.torch.version, "hip", hip)
    model = SimpleNamespace(_yue2_fp8_originals={"weight": 1} if fp8 else {})
    assert sampling.graph_fallback_reason(torch.device(device), model) == reason


def test_profile_and_environment_change_result_identity(runtime, tmp_path, monkeypatch):
    pipe = object.__new__(pipeline.YuE2Pipeline)
    pipe.generation_config = GenerationConfig()
    pipe.backend, pipe.quantization = "torch", "none"
    pipe.vae_core_frames, pipe.memory_budget_gib, pipe.offload_ar = 1024, 32, False
    pipe.device, pipe.runtime_sha256 = torch.device("cuda:1"), "source-identity"
    pipe.vae_dir = tmp_path
    (tmp_path / "config.json").write_text(json.dumps({"release_variant": "listening"}))
    request = SongRequest("piano", "original lyric", seed=42)
    configs = []
    for profile, env in [(None, "0"), ("windows-gfx1151", "0"), ("windows-gfx1151", "1")]:
        pipe.rocm_profile = profile
        monkeypatch.setenv(rocm.AOTRITON_ENV, env)
        config = pipe.effective_config(request)
        assert config["generation"] == GenerationConfig().to_dict()
        assert (config["model_dtype"], config["vae_dtype"]) == ("bfloat16", "float32")
        assert config["validation_status"] == "unvalidated"
        configs.append(config)
    assert len({identity(config) for config in configs}) == 3
    assert configs[-1]["rocm"]["vae_convolution"] == "torch-native"
    assert configs[-1]["rocm"]["attention"] == "sdpa-auto"
    assert configs[-1]["rocm"]["experimental_aotriton"] == "1"
    runtime[0].version.hip = None
    assert "rocm" not in pipe.effective_config(request)


@pytest.mark.parametrize("profile", [None, "windows-gfx1151"])
def test_cli_profile_reaches_pipeline(monkeypatch, profile):
    captured = {}

    def load(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pipeline.YuE2Pipeline, "from_pretrained", load)
    argv = ["generate"] + (["--rocm-profile", profile] if profile else [])
    cli.get_pipe(cli.parser().parse_args(argv))
    assert captured["rocm_profile"] == profile
    assert captured["backend"] == "torch" and captured["quantization"] == "none"
