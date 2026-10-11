import pytest
import torch
from yue2.modeling_vae import YuE2VAE
from test_vae import tiny_config

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
