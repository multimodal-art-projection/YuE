"""Graph decode equations on CPU; optional CUDA capture parity on a real GPU."""
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from yue2.cuda_graph import GraphAR
from yue2.modeling_yue2 import StaticKVCache, YuE2Config, YuE2ForCausalLM


@pytest.fixture
def model():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(146)
        model = YuE2ForCausalLM(YuE2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=4, vocab_size=32,
            max_position_embeddings=64, max_latent_frames=64)).eval()
    yield model
    torch.set_num_threads(previous)


def reference_prefill(model, prefixes, budget):
    weight, config = model.model.embed_tokens.weight, model.config
    caches, logits = [], []
    for prefix in prefixes:
        cache = StaticKVCache(config.num_hidden_layers, 1, config.num_key_value_heads,
                             len(prefix) + budget, config.head_dim, weight.dtype, weight.device)
        out = model(torch.tensor([prefix], device=weight.device), past_key_values=cache,
                    use_cache=True, logits_to_keep=1)
        caches.append(cache)
        logits.append(out.logits[:, -1])
    return caches, torch.cat(logits)


@pytest.mark.parametrize("prefixes", [[[2, 3, 4]], [[2, 3, 4, 5], [6]]])
@pytest.mark.parametrize("fused", [False, True])
def test_graph_equations_match_original_static_cache_and_only_use_ar_experts(model, prefixes, fused):
    with torch.inference_mode(), ExitStack() as stack:
        forbidden = [stack.enter_context(patch.object(layer.nar_self_attn, "project_qkv", side_effect=AssertionError("NAR used")))
                     for layer in model.model.layers]
        original, expected = reference_prefill(model, prefixes, 5)
        graph = GraphAR(model, prefixes, 5, capture=False, fuse_projections=fused)
        torch.testing.assert_close(graph.prefill(), expected, atol=1e-7, rtol=1e-5)
        pointers = [value.data_ptr() for value in graph.keys + graph.values]
        for index, token in enumerate([7, 8, 9, 10]):
            expected = torch.cat([model(torch.tensor([[token]]), past_key_values=cache,
                                       use_cache=True, logits_to_keep=1).logits[:, -1] for cache in original])
            actual = graph.step(token)
            torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
            assert graph.positions.tolist() == [len(prefix) + index + 1 for prefix in prefixes]
            assert [value.data_ptr() for value in graph.keys + graph.values] == pointers
        assert all(mock.call_count == 0 for mock in forbidden)
        with pytest.raises(ValueError, match="budget"):
            graph.step(11)
        graph.close()
        assert graph.keys == [] and graph.positions is None


def test_mask_hides_unused_slots_and_prevents_cross_branch_visibility(model):
    prefixes = [[2, 3, 4, 5], [6]]
    with torch.inference_mode():
        baseline = GraphAR(model, prefixes, 5, capture=False)
        changed = GraphAR(model, prefixes, 5, capture=False)
        baseline.prefill()
        changed.prefill()
        for keys, values in zip(changed.keys, changed.values):
            for branch, prefix in enumerate(prefixes):
                keys[branch, len(prefix)+1:].fill_(1000)
                values[branch, len(prefix)+1:].fill_(-1000)
        torch.testing.assert_close(changed.step(7), baseline.step(7), atol=0, rtol=0)
        # Perturb only branch 0's visible cache: branch 1 remains unchanged.
        for values in changed.values:
            values[0, :2].add_(1)
        old, new = baseline.step(8), changed.step(8)
        torch.testing.assert_close(new[1], old[1], atol=0, rtol=0)
        assert not torch.equal(new[0], old[0])


def test_cfg_accepts_only_one_shared_token_and_validates_lifecycle(model):
    graph = GraphAR(model, [[2, 3], [4]], 2, capture=False)
    with pytest.raises(RuntimeError, match="prefill"):
        graph.step(1)
    graph.prefill()
    with pytest.raises(RuntimeError, match="exactly once"):
        graph.prefill()
    with pytest.raises(ValueError, match="one shared"):
        graph.step(torch.tensor([2, 3]))
    with pytest.raises(ValueError, match="vocabulary"):
        graph.step(100)
    graph.step(torch.tensor([[7]]))
    assert graph.tokens.tolist() == [[7], [7]]
    graph.close()
    with pytest.raises(RuntimeError, match="closed"):
        graph.step(1)


@pytest.mark.parametrize("prefixes,budget", [([[]], 2), ([[1.2]], 2), ([[33]], 2),
                                             ([[1]] * 3, 2), ([[1]], 0), ([[1] * 60], 5)])
def test_invalid_capacity_is_not_silently_shortened(model, prefixes, budget):
    with pytest.raises(ValueError):
        GraphAR(model, prefixes, budget, capture=False)


def test_default_capture_rejects_cpu(model):
    with pytest.raises(ValueError, match="CUDA"):
        GraphAR(model, [[1]], 2)


def test_quantized_model_requires_explicit_eager_path(model):
    object.__setattr__(model, "_yue2_fp8_originals", {"original": object()})
    with pytest.raises(ValueError, match="eager for FP8"):
        GraphAR(model, [[1]], 2, capture=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA capture requires an actual allocated GPU")
@pytest.mark.parametrize("prefixes", [[[2, 3, 4]], [[2, 3, 4, 5], [6]]])
@pytest.mark.parametrize("backend", ["auto", "cudnn"])
@pytest.mark.parametrize("fused", [False, True])
def test_real_cuda_graph_bfloat16_parity(model, prefixes, backend, fused):
    # Head size 8 exercises fused CUDA paths, as does the real head size 128.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(146)
        model = YuE2ForCausalLM(YuE2Config(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=8, vocab_size=32,
            max_position_embeddings=64, max_latent_frames=64)).eval().to(device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        original, expected = reference_prefill(model, prefixes, 5)
        graph = GraphAR(model, prefixes, 5, attention_backend=backend, fuse_projections=fused)
        torch.testing.assert_close(graph.prefill(), expected, atol=0, rtol=0)
        assert graph.graph is not None
        assert graph.attention_backend == ("flash" if backend == "auto" else "cudnn")
        for keys, values in zip(graph.keys, graph.values):
            for branch, prefix in enumerate(prefixes):
                keys[branch, len(prefix)+1:].fill_(1000)
                values[branch, len(prefix)+1:].fill_(-1000)
        for token in [7, 8, 9, 10]:
            expected = torch.cat([model(torch.tensor([[token]], device="cuda"), past_key_values=cache,
                                       use_cache=True, logits_to_keep=1).logits[:, -1] for cache in original])
            actual = graph.step(token).clone()
            torch.testing.assert_close(actual, expected, atol=.002, rtol=.02)
        graph.close()
