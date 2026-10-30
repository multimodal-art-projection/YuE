import pytest
import torch
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from yue2.protocol import Sampling, EOD, ABC_END, MUSIC_END, CODEC_OFFSET, CODEC_SIZE, VOCAB_SIZE
from yue2.sampling import generate_tokens


def native_model():
    torch.set_num_threads(1)
    torch.manual_seed(317)
    return YuE2ForCausalLM(
        YuE2Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=VOCAB_SIZE,
            max_position_embeddings=16,
            max_latent_frames=16,
        )
    ).eval()


@pytest.mark.parametrize("phase", ["abc", "semantic"])
def test_greedy_nonfinite_model_logits_never_become_tokens(phase):
    model = native_model()
    with torch.no_grad():
        model.lm_head.weight.fill_(float("nan"))
    events = []
    config = Sampling(temperature=0, min_tokens=0, max_tokens=2)
    with pytest.raises(FloatingPointError, match="finite"):
        generate_tokens(
            model,
            [1, 2],
            config,
            41,
            phase,
            use_cuda_graph=False,
            on_token=lambda stage, token: events.append(token),
        )
    assert events == []


@pytest.mark.parametrize("phase", ["abc", "semantic"])
def test_valid_greedy_model_preserves_phase_vocabulary_and_repeatability(phase):
    model = native_model()
    config = Sampling(temperature=0, min_tokens=0, max_tokens=2)
    first, _, truncated = generate_tokens(model, [1, 2], config, 41, phase, use_cuda_graph=False)
    second, _, again = generate_tokens(model, [1, 2], config, 41, phase, use_cuda_graph=False)
    assert first == second and truncated == again
    assert first
    if phase == "abc":
        assert all(0 <= token < EOD for token in first)
    else:
        assert all(CODEC_OFFSET <= token < CODEC_OFFSET + CODEC_SIZE for token in first)
