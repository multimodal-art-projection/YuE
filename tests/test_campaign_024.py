import pytest
import torch
from yue2.modeling_vae import YuE2VAE, YuE2VAEConfig
from test_vae import tiny_config

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
