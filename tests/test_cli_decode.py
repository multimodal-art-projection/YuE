import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import soundfile as sf
import torch
from yue2.modeling_vae import YuE2VAE
from test_vae import tiny_config


@pytest.fixture
def native(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(829)
    model = YuE2VAE(tiny_config()).eval()
    vae = tmp_path / "vae"
    model.save_pretrained(vae)
    latent = torch.randn(1, 2, 3)
    expected = model.decode(latent)[0].transpose(0, 1).detach().numpy().clip(-1, 1)
    return vae, latent.numpy(), expected


def command(tmp_path, vae, latent, output):
    boundary = tmp_path / "no-ar"
    boundary.mkdir(exist_ok=True)
    (boundary / "sitecustomize.py").write_text(
        "import sys\nclass Boundary:\n    def find_spec(self, fullname, path=None, target=None):\n        if fullname in {'yue2.pipeline','yue2.modeling_yue2','yue2.tokenization_yue2'}: raise RuntimeError('decoder loaded AR runtime')\nsys.meta_path.insert(0,Boundary())\n"
    )
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            (str(boundary), str(Path(__file__).resolve().parents[1] / "src"))
        ),
        OMP_NUM_THREADS="1",
    )
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "yue2.cli",
            "decode",
            str(latent),
            "--vae",
            str(vae),
            "--device",
            "cpu",
            "--offline",
            "--output",
            str(output),
        ],
        env=env,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("layout", ["channels-first", "frames-first"])
def test_decoder_only_cli_matches_native_full_waveform(tmp_path, native, layout):
    vae, latent, expected = native
    source = tmp_path / "latent.npy"
    np.save(source, latent if layout == "channels-first" else latent[0].T)
    output = tmp_path / "decoded.wav"
    result = command(tmp_path, vae, source, output)
    assert result.returncode == 0, result.stderr
    audio, rate = sf.read(output, dtype="float32")
    np.testing.assert_array_equal(audio, expected)
    record = json.loads(result.stdout)
    assert record["operation"] == "decode_latents" and record["sample_rate"] == rate == 48000
    assert record["audio_seconds"] == len(expected) / rate
    assert record["decoder_only"] is True and record["quality_checked"] is False
    assert (
        record["weights"]["files"]["model.safetensors"]["bytes"]
        == (vae / "model.safetensors").stat().st_size
    )


def test_decoder_cli_preserves_existing_audio_before_loading_weights(tmp_path):
    output = tmp_path / "audio.wav"
    output.write_bytes(b"keep existing recording")
    result = command(tmp_path, tmp_path / "missing-vae", tmp_path / "missing.npy", output)
    assert result.returncode != 0 and "already exists" in result.stderr
    assert output.read_bytes() == b"keep existing recording"


def test_decoder_cli_refuses_nonfinite_latents_without_creating_audio(tmp_path, native):
    vae, latent, _ = native
    latent[0, 0, 0] = np.nan
    source = tmp_path / "latent.npy"
    np.save(source, latent)
    output = tmp_path / "audio.wav"
    result = command(tmp_path, vae, source, output)
    assert result.returncode != 0 and "finite" in result.stderr
    assert not output.exists()


def test_existing_decoder_only_python_api_control(native):
    vae, latent, expected = native
    model = YuE2VAE.from_pretrained(vae, decoder_only=True, device="cpu", local_files_only=True)
    assert model.decoder_only and not hasattr(model, "encoder")
    actual = model.decode_tiled(latent)[0].transpose(0, 1).numpy().clip(-1, 1)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("invalid", ["nan", "empty", "integer", "rank", "archive"])
def test_bad_latent_file_is_rejected_before_resolving_model(tmp_path, monkeypatch, invalid):
    from yue2 import cli, storage

    latent = np.zeros((3, 64), dtype=np.float32)
    if invalid == "nan":
        latent[0, 0] = np.nan
    elif invalid == "empty":
        latent = latent[:0]
    elif invalid == "integer":
        latent = latent.astype(np.int32)
    elif invalid == "rank":
        latent = latent.reshape(-1)
    source = tmp_path / ("latent.npz" if invalid == "archive" else "latent.npy")
    if invalid == "archive":
        np.savez(source, latent=latent)
    else:
        np.save(source, latent)
    monkeypatch.setattr(
        storage, "resolve_model", lambda *a, **k: pytest.fail("model resolution ran")
    )
    output = tmp_path / "audio.wav"
    with pytest.raises(ValueError):
        cli.main(["decode", str(source), "--vae", "example/decoder", "--output", str(output)])
    assert not output.exists()


def test_decoder_refuses_nonfinite_output_and_releases_model(tmp_path, native, monkeypatch):
    from yue2 import cli

    vae, latent, _ = native
    source, output = tmp_path / "latent.npy", tmp_path / "bad.wav"
    np.save(source, latent)
    moved = []

    class Decoder:
        def decode_tiled(self, *args, **kwargs):
            return torch.full((1, 2, 8), float("nan"))

        def to(self, device):
            moved.append(device)

    monkeypatch.setattr(YuE2VAE, "from_pretrained", lambda *a, **k: Decoder())
    with pytest.raises(ValueError, match="non-finite audio"):
        cli.main(["decode", str(source), "--vae", str(vae), "--offline", "--output", str(output)])
    assert moved == ["cpu"]
    assert not output.exists()
