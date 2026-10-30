


import numpy as np
import pytest
import torch
from yue2.modeling_vae import YuE2VAE, YuE2VAEConfig
from yue2.pipeline import YuE2Pipeline

def decoder_fixture(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(913)
    common = dict(channels=1, c_mults=[1]*6, strides=[2,2,4,4,5,6], use_snake=True)
    config = YuE2VAEConfig(encoder_config=dict(common, in_channels=2, latent_dim=128),
                          decoder_config=dict(common, out_channels=2, latent_dim=64,
                                              snake_type='vanilla', final_tanh=False), latent_dim=64)
    model = YuE2VAE(config).eval()
    model.save_pretrained(tmp_path / 'vae')
    pipe = YuE2Pipeline.__new__(YuE2Pipeline)
    pipe.low_vram = False
    pipe.device = torch.device('cpu')
    pipe._model = pipe._vae = None
    pipe.vae_dir = tmp_path / 'vae'
    pipe.vae_core_frames = 2
    pipe.progress = False
    return pipe, model

@pytest.mark.parametrize('invalid', ['shape', 'empty', 'nan', 'inf'])
def test_invalid_latents_fail_before_loading_any_decoder(tmp_path, monkeypatch, invalid):
    pipe, model = decoder_fixture(tmp_path)
    calls = []
    original = YuE2VAE.from_pretrained
    def load(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)
    monkeypatch.setattr(YuE2VAE, 'from_pretrained', load)
    value = torch.zeros(1, 64, 3)
    if invalid == 'shape': value = torch.zeros(1, 63, 3)
    elif invalid == 'empty': value = torch.zeros(1, 64, 0)
    else: value[0,0,0] = float(invalid)
    with pytest.raises(ValueError): pipe.decode(value, full=True)
    assert calls == []
    assert pipe._vae is None

@pytest.mark.parametrize('layout', ['frames', 'batched'])
@torch.inference_mode()
def test_valid_decode_layouts_keep_exact_native_audio(tmp_path, layout):
    pipe, model = decoder_fixture(tmp_path)
    z = torch.randn(1, 64, 3) * 0.1
    expected = model.decode(z)[0].float().clamp(-1, 1).T.contiguous().numpy()
    value = z[0].T.numpy() if layout == 'frames' else z
    actual = pipe.decode(value, full=True)
    np.testing.assert_array_equal(actual, expected)
    assert pipe._vae is not None
    assert all(p.device.type == 'cpu' for p in pipe._vae.parameters())
