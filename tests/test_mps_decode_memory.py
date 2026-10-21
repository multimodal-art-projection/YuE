"""MPS decode releases the allocator's cached buffers while the KV length grows."""
import pytest
import torch

from yue2 import sampling
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from yue2.protocol import Sampling

pytestmark = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="requires MPS")


def test_decode_empties_the_mps_cache_every_32_tokens(monkeypatch):
    torch.manual_seed(5)
    config = YuE2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                        num_key_value_heads=1, head_dim=8, vocab_size=184704, max_position_embeddings=128)
    model = YuE2ForCausalLM(config).eval().to("mps")
    calls = []
    monkeypatch.setattr(torch.mps, "empty_cache", lambda: calls.append(True))
    tokens, timing, truncated = sampling.generate_tokens(
        model, [151643, 151644, 151645], Sampling(1.0, 0.95, 100, 1.2, 50, 70, 70), 3, "semantic")
    assert len(tokens) == 70 and truncated
    assert len(calls) == 70 // 32
