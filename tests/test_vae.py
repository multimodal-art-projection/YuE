"""CPU topology, exact boundary and portable serialization checks."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
import pytest
import torch
from transformers import AutoModel
from yue2.modeling_vae import YuE2VAE, YuE2VAEConfig
from huggingface_hub.file_download import _create_symlink
from yue2.modeling_vae import YuE2VAE


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


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


@pytest.mark.parametrize('decoder_only', [False,True])
@pytest.mark.parametrize('mode', ['parent','absolute','nested-parent'])
def test_indexed_shard_rejects_absolute_and_parent_filenames(tmp_path, mode, decoder_only):
    torch.set_num_threads(1)
    directory = tmp_path/'snapshot'
    original = YuE2VAE(tiny_config()); original.save_pretrained(directory)
    outside = tmp_path/'outside.safetensors'
    (directory/'model.safetensors').replace(outside)
    if mode == 'parent': name = '../outside.safetensors'
    elif mode == 'absolute': name = str(outside)
    else:
        (directory/'shards').mkdir()
        name = 'shards/../../outside.safetensors'
    (directory/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{key:name for key in original.state_dict()}}))
    with pytest.raises(ValueError,match='shard path'):
        YuE2VAE.from_pretrained(directory,decoder_only=decoder_only,local_files_only=True)

@pytest.mark.parametrize('decoder_only', [False,True])
def test_safe_nested_indexed_snapshot_preserves_exact_decoder_output(tmp_path,decoder_only):
    torch.set_num_threads(1); torch.manual_seed(247)
    directory = tmp_path/'snapshot'
    original = YuE2VAE(tiny_config()); original.save_pretrained(directory)
    (directory/'shards').mkdir()
    (directory/'model.safetensors').replace(directory/'shards/weights.safetensors')
    (directory/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{key:'shards/weights.safetensors' for key in original.state_dict()}}))
    loaded = YuE2VAE.from_pretrained(directory,decoder_only=decoder_only,local_files_only=True)
    latent = torch.randn(1,2,3)
    assert torch.equal(original.decode(latent), loaded.decode(latent))


@pytest.mark.parametrize('decoder_only', [False, True])
def test_standard_hf_cache_snapshot_can_load_indexed_vae(tmp_path, decoder_only):
    torch.set_num_threads(1)
    torch.manual_seed(247)
    repo_cache = tmp_path/'models--owner--vae'
    snapshot = repo_cache/'snapshots'/'abc123'
    model = YuE2VAE(tiny_config())
    model.save_pretrained(snapshot)
    blobs = repo_cache/'blobs'; blobs.mkdir()
    blob = blobs/'weights-content-hash'
    shard = snapshot/'model-00001-of-00001.safetensors'
    (snapshot/'model.safetensors').replace(blob)
    _create_symlink(str(blob), str(shard), new_blob=False)
    (snapshot/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {name: shard.name for name in model.state_dict()}}))
    assert shard.is_symlink()
    loaded = YuE2VAE.from_pretrained(snapshot, decoder_only=decoder_only, local_files_only=True)
    latent = torch.randn(1,2,3)
    assert torch.equal(model.decode(latent), loaded.decode(latent))

@pytest.mark.parametrize('decoder_only', [False, True])
@pytest.mark.parametrize('name', [r'\outside.safetensors', r'C:outside.safetensors', r'..\outside.safetensors'])
def test_portable_indexed_shard_filenames_reject_windows_root_drive_and_parent(tmp_path, name, decoder_only):
    if Path(name).name != name:
        pytest.skip('POSIX literal-backslash fixture exercises portable filename validation')
    torch.set_num_threads(1)
    directory = tmp_path/'snapshot'
    original = YuE2VAE(tiny_config()); original.save_pretrained(directory)
    (directory/'model.safetensors').replace(directory/name)
    (directory/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{key:name for key in original.state_dict()}}))
    with pytest.raises(ValueError,match='shard path'):
        YuE2VAE.from_pretrained(directory,decoder_only=decoder_only,local_files_only=True)


@pytest.mark.parametrize('sample', [False, True])
def test_vae_encoder_preserves_fp32_posterior_under_cpu_autocast(sample):
    torch.set_num_threads(1)
    torch.manual_seed(249)
    model = YuE2VAE(tiny_config())
    audio = torch.randn(1, 2, 3 * 1920)
    expected, expected_info = model.encode(audio, sample=sample, return_info=True,
                                          generator=torch.Generator().manual_seed(739))
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual, info = model.encode(audio, sample=sample, return_info=True,
                                    generator=torch.Generator().manual_seed(739))
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)
    for name in ('mean', 'scale', 'stdev'):
        assert info[name].dtype == torch.float32
        assert torch.equal(info[name], expected_info[name])

def test_vae_decoder_already_preserves_fp32_under_cpu_autocast():
    torch.set_num_threads(1)
    torch.manual_seed(924)
    model = YuE2VAE(tiny_config())
    latent = torch.randn(1, 2, 3)
    expected = model.decode(latent)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = model.decode(latent)
    assert actual.dtype == torch.float32
    assert torch.equal(actual, expected)


@pytest.mark.parametrize('inconsistent', ['posterior','encoder-ratio','encoder-channels','decoder-channels'])
def test_vae_configuration_rejects_incompatible_encode_decode_contracts(inconsistent):
    torch.set_num_threads(1)
    original = tiny_config()
    encoder = dict(original.encoder_config)
    decoder = dict(original.decoder_config)
    if inconsistent == 'posterior': encoder['latent_dim'] = 6
    elif inconsistent == 'encoder-ratio': encoder['strides'] = [1,2,4,4,5,6]
    elif inconsistent == 'encoder-channels': encoder['in_channels'] = 3
    else: decoder['out_channels'] = 3
    with pytest.raises(ValueError,match='Encoder|Audio channels'):
        config = YuE2VAEConfig(encoder_config=encoder,decoder_config=decoder,latent_dim=2)
        model = YuE2VAE(config)
        model(torch.randn(1,2,3*1920))

@torch.inference_mode()
def test_consistent_tiny_topology_encodes_and_decodes_exact_native_dimensions(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(123)
    model = YuE2VAE(tiny_config())
    audio = torch.randn(1,2,3*1920)
    latent = model.encode(audio)
    waveform = model.decode(latent)
    assert latent.shape == (1,2,3)
    assert waveform.shape == (1,2,3*1920-64)
    assert torch.isfinite(waveform).all()
    model.save_pretrained(tmp_path / 'vae')
    restored = YuE2VAE.from_pretrained(tmp_path / 'vae',local_files_only=True)
    assert torch.equal(restored.encode(audio),latent)
    assert torch.equal(restored.decode(latent),waveform)
