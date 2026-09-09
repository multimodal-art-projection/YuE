"""CPU topology, exact boundary and portable serialization checks."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from yue2.modeling_vae import YuE2VAE, YuE2VAEConfig


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tiny_config():
    # Same six-stage strides/padding/activation topology, reduced channels.
    common = dict(channels=1, c_mults=[1, 1, 1, 1, 1, 1],
                  strides=[2, 2, 4, 4, 5, 6], use_snake=True)
    return YuE2VAEConfig(
        encoder_config=dict(common, in_channels=2, latent_dim=4),
        decoder_config=dict(common, out_channels=2, latent_dim=2,
                            snake_type="vanilla", final_tanh=False), latent_dim=2)


def test_real_configuration_keys_and_lengths():
    # Meta construction checks the real schema without allocating 0.53GB.
    with torch.device("meta"):
        model = YuE2VAE(YuE2VAEConfig())
    state = model.state_dict()
    assert model.config.decode_core_frames == 1024
    assert model.config.decode_halo_frames == 16
    assert len(state) == 435
    assert sum(x.numel() for x in state.values()) == 132616130
    assert sum(k.startswith("encoder.") for k in state) == 218
    assert sum(k.startswith("decoder.") for k in state) == 217
    for frames in [1, 2, 16, 256, 257, 9000]:
        assert model.natural_output_length(frames) == 1920 * frames - 64
    assert model.required_halo(256) <= 16


@pytest.mark.parametrize("frames,core", [(1, 1), (15, 7), (17, 8), (33, 16),
                                         (65, 32), (257, 256)])
def test_tiled_matches_full_without_crossfade(frames, core):
    torch.manual_seed(231)
    model = YuE2VAE(tiny_config())
    latent = torch.randn(1, 2, frames)
    expected = model.decode(latent)
    actual = model.decode_tiled(latent, core_frames=core)
    assert actual.shape == (1, 2, frames * 1920 - 64)
    assert actual.device.type == "cpu"
    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
    for seam in range(core * 1920, actual.shape[-1], core * 1920):
        torch.testing.assert_close(actual[..., seam-128:seam+128],
                                   expected[..., seam-128:seam+128],
                                   rtol=3e-5, atol=3e-6)


def test_tiling_rejects_insufficient_halo_and_bad_latents():
    model = YuE2VAE(tiny_config())
    with pytest.raises(ValueError, match="halo"):
        model.decode_tiled(torch.zeros(1, 2, 32), halo_frames=0)
    for latent in [torch.zeros(1, 3, 32), torch.zeros(1, 2, 0),
                   torch.full((1, 2, 8), float("nan"))]:
        with pytest.raises(ValueError):
            model.decode_tiled(latent)


@pytest.mark.parametrize("frames,core", [(1, 1), (17, 8), (32, 16)])
def test_tiled_progress_counts_copied_tiles_and_preserves_output(frames, core):
    torch.manual_seed(751)
    model = YuE2VAE(tiny_config())
    latent = torch.randn(1, 2, frames)
    expected = model.decode_tiled(latent, core, None, "cpu")
    events = []
    rng_state = torch.random.get_rng_state().clone()
    with patch.object(model, "decode", wraps=model.decode) as decode:
        def progress(completed, total):
            assert decode.call_count == completed
            assert type(completed) is int and type(total) is int
            events.append((completed, total))

        # decode_audio must forward the callback to the tiled implementation.
        actual = model.decode_audio(latent, core_frames=core, on_progress=progress)
    total = (frames + core - 1) // core
    assert events == [(index, total) for index in range(1, total + 1)]
    assert torch.equal(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_tiled_progress_exception_stops_further_tiles():
    model = YuE2VAE(tiny_config())
    events = []

    def progress(completed, total):
        events.append((completed, total))
        raise RuntimeError("progress handler failed")

    with patch.object(model, "decode", wraps=model.decode) as decode:
        with pytest.raises(RuntimeError, match="progress handler failed"):
            model.decode_tiled(torch.zeros(1, 2, 17), core_frames=8, on_progress=progress)
    assert events == [(1, 3)]
    assert decode.call_count == 1


def test_roundtrip_and_decoder_only(tmp_path):
    torch.manual_seed(451)
    original = YuE2VAE(tiny_config())
    original.save_pretrained(tmp_path)
    loaded = YuE2VAE.from_pretrained(tmp_path)
    decoder = YuE2VAE.from_pretrained(tmp_path, decoder_only=True)
    assert len(loaded.state_dict()) == 435
    assert len(decoder.state_dict()) == 217
    assert not hasattr(decoder, "encoder")
    for key, tensor in original.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[key])
    latent = torch.randn(1, 2, 19)
    assert torch.equal(original.decode(latent), decoder.decode(latent))
    with pytest.raises(RuntimeError, match="Encoder not loaded"):
        decoder.encode(torch.zeros(1, 2, 1920))
    with pytest.raises(ValueError, match="complete"):
        decoder.save_pretrained(tmp_path / "incomplete")
    with pytest.raises(ValueError, match="FP32"):
        YuE2VAE.from_pretrained(tmp_path, torch_dtype=torch.float16)
    assert (tmp_path / "modeling_vae.py").exists()
    assert json.loads((tmp_path / "config.json").read_text())["auto_map"]["AutoModel"]


def test_hf_auto_model(tmp_path):
    YuE2VAE(tiny_config()).save_pretrained(tmp_path)
    loaded = AutoModel.from_pretrained(tmp_path, trust_remote_code=True,
                                       decoder_only=True, device="cpu")
    assert loaded.config.model_type == "yue2_vae"
    assert loaded.decode_tiled(torch.zeros(1, 2, 3)).shape == (1, 2, 5696)


def test_encode_posterior_seed(tmp_path):
    model = YuE2VAE(tiny_config())
    audio = torch.randn(1, 2, 3 * 1920)
    mean, info = model.encode(audio, return_info=True)
    assert torch.equal(mean, info["mean"])
    a = model.encode(audio, sample=True, generator=torch.Generator().manual_seed(79))
    b = model.encode(audio, sample=True, generator=torch.Generator().manual_seed(79))
    assert torch.equal(a, b)
    assert not torch.equal(mean, a)
