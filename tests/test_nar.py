"""Portable CPU checks against the original dense AR–NAR forward equations."""
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from yue2 import nar
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM


@pytest.fixture
def model():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng():
        torch.manual_seed(173)
        config = YuE2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=4,
                            vocab_size=32, max_position_embeddings=128,
                            latent_dim=64, vae_latent_dim=64, max_latent_frames=128)
        value = YuE2ForCausalLM(config).eval()
    yield value
    torch.set_num_threads(previous_threads)


def dense_velocity(model, chunk, state, raw_t):
    ar_length, total = len(chunk.ar_tokens), len(chunk.ar_tokens) + len(state) + 2
    tokens = torch.tensor([chunk.ar_tokens + [0] * (len(state) + 2)])
    ar_mask = torch.arange(total)[None] < ar_length
    content_mask = (torch.arange(total)[None] > ar_length) & (torch.arange(total)[None] < total - 1)
    return model.nar_velocity(tokens, ar_mask, ~ar_mask, content_mask, state, raw_t,
                              nar_cond_end=chunk.nar_cond_end)


@pytest.mark.parametrize("causal", [False, True])
def test_query_tiling_preserves_all_keys_and_gqa(causal):
    generator = torch.Generator().manual_seed(719)
    q = torch.randn(7, 4, 8, generator=generator)
    k = torch.randn(7 if causal else 11, 2, 8, generator=generator)
    v = torch.randn(k.shape, generator=generator)
    expected = F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.repeat_interleave(2, 1).transpose(0, 1)[None],
        v.repeat_interleave(2, 1).transpose(0, 1)[None], is_causal=causal,
    )[0].transpose(0, 1)
    with patch.object(nar.F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as call:
        result = nar.attention(q, k, v, causal=causal, backend="math", query_chunk_size=3)
    assert len(call.call_args_list) == 3
    assert all(row.args[0].shape[-2] <= 3 for row in call.call_args_list)
    assert [row.args[1].shape[-2] for row in call.call_args_list] == ([3, 6, 7] if causal else [11, 11, 11])
    torch.testing.assert_close(result, expected, atol=4e-7, rtol=4e-6)


def test_noise_is_one_cpu_float32_draw_before_original_chunks():
    state = torch.random.get_rng_state().clone()
    chunks = nar.song_chunks([2, 3], list(range(11)), 812, context=15)
    expected = torch.randn((11, 64), generator=torch.Generator(device="cpu").manual_seed(812))
    assert [len(chunk.noise) for chunk in chunks] == [5, 5, 1]
    assert all(chunk.noise.dtype == torch.float32 and chunk.noise.device.type == "cpu" for chunk in chunks)
    assert len({chunk.noise.untyped_storage().data_ptr() for chunk in chunks}) == 1
    torch.testing.assert_close(torch.cat([chunk.noise for chunk in chunks]), expected, atol=0, rtol=0)
    assert chunks[1].ar_tokens == [2, 3] + [nar.CODEC_OFFSET + i for i in range(5, 10)] + [nar.MUSIC_END]
    torch.testing.assert_close(torch.random.get_rng_state(), state, atol=0, rtol=0)


@pytest.mark.parametrize("prefix,codec,seed,context", [
    ([], [1], 2, 15), ([1.0], [1], 2, 15), ([1], [True], 2, 15),
    ([1], [-1], 2, 15), ([1], [nar.CODEC_SIZE], 2, 15),
    ([1], [1], 2.5, 15), ([1], [1], 2, nar.CONTEXT + 1),
])
def test_invalid_protocol_inputs_are_rejected(prefix, codec, seed, context):
    with pytest.raises(ValueError):
        nar.song_chunks(prefix, codec, seed, context)


@pytest.mark.parametrize("visible", [0, 2])
def test_cached_velocity_matches_dense_original_forward(model, visible):
    noise = torch.randn((5, 64), generator=torch.Generator().manual_seed(42))
    chunk = nar.Chunk([2, 3, 4, 5], noise, nar_cond_end=visible)
    engine = nar.CachedNAR(model, chunk, query_chunk_size=3)
    for raw_t in [20.0, 0.0, -2.3]:
        expected = dense_velocity(model, chunk, noise, raw_t)
        actual = engine.velocity(noise, raw_t)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-5)
    assert all(len(k) == (visible or 4) for k, _ in engine.cache)
    engine.close()


def test_midpoint_solution_matches_dense_original_solver_and_prefills_once(model):
    chunk = nar.Chunk([2, 3, 4, 5], torch.randn((3, 64), generator=torch.Generator().manual_seed(381)))
    with ExitStack() as stack:
        ar_projections = [stack.enter_context(patch.object(layer.self_attn, "project_qkv", wraps=layer.self_attn.project_qkv))
                          for layer in model.model.layers]
        engine = nar.CachedNAR(model, chunk)
        actual = engine.solve(steps=4)
        assert [call.call_count for call in ar_projections] == [1, 1]
    expected, dt = chunk.noise.clone(), 1 / 4
    for step in range(4):
        t = 1 - step * dt
        raw = torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20).item()
        first = dense_velocity(model, chunk, expected, raw)
        midpoint = expected - first * (dt / 2)
        raw_mid = torch.logit(torch.tensor(t - dt / 2, dtype=torch.float64)).clamp(-20, 20).item()
        expected = expected - dense_velocity(model, chunk, midpoint, raw_mid) * dt
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=2e-5)
    assert actual.dtype == torch.float32 and actual.device.type == "cpu" and not actual.requires_grad


def test_default_is_32_midpoint_steps(model):
    noise = torch.zeros((2, 64))
    engine = nar.CachedNAR(model, nar.Chunk([2, 3], noise))
    with patch.object(engine, "velocity", return_value=torch.ones_like(noise)) as velocity:
        actual = engine.solve()
    assert velocity.call_count == 64
    torch.testing.assert_close(actual, -torch.ones_like(noise), atol=0, rtol=0)
    assert velocity.call_args_list[0].args[1] == 20.0


def test_midpoint_progress_counts_complete_steps_without_changing_output(model):
    noise = torch.randn((3, 64), generator=torch.Generator().manual_seed(391))
    engine = nar.CachedNAR(model, nar.Chunk([2, 3], noise))
    expected = engine.solve(4, None)
    events = []
    with patch.object(engine, "velocity", wraps=engine.velocity) as velocity:
        def progress(completed, total):
            assert velocity.call_count == completed * 2
            assert type(completed) is int and type(total) is int
            events.append((completed, total))

        actual = engine.solve(4, None, progress)
    assert events == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert torch.equal(actual, expected)


def test_midpoint_progress_exception_stops_further_steps(model):
    engine = nar.CachedNAR(model, nar.Chunk([2, 3], torch.zeros(2, 64)))
    events = []

    def progress(completed, total):
        events.append((completed, total))
        raise RuntimeError("progress handler failed")

    with patch.object(engine, "velocity", wraps=engine.velocity) as velocity:
        with pytest.raises(RuntimeError, match="progress handler failed"):
            engine.solve(4, on_progress=progress)
    assert events == [(1, 4)]
    assert velocity.call_count == 2


def test_synthesis_solves_original_chunks_serially_and_is_repeatable(model, monkeypatch):
    # Keep the architecture tiny while testing actual token construction and
    # original chunk boundaries; token-ID values do not affect memory ordering.
    monkeypatch.setattr(nar, "CODEC_OFFSET", 8)
    monkeypatch.setattr(nar, "MUSIC_END", 7)
    chunks = nar.song_chunks([2, 3], [1] * 11, 42, context=15)
    expected = torch.cat([nar.CachedNAR(model, chunk).solve(steps=2) for chunk in chunks])
    events = []
    original = nar.CachedNAR

    class Tracked(original):
        def __init__(self, *args, **kwargs):
            events.append("prefill")
            super().__init__(*args, **kwargs)

        def close(self):
            events.append("release")
            super().close()

    monkeypatch.setattr(nar, "CachedNAR", Tracked)
    actual = nar.synthesize(model, [2, 3], [1] * 11, 42, steps=2, context=15, offload_ar=True)
    assert events == ["prefill", "release"] * 3
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_synthesis_progress_accumulates_all_chunks_and_preserves_rng(model, monkeypatch):
    monkeypatch.setattr(nar, "CODEC_OFFSET", 8)
    monkeypatch.setattr(nar, "MUSIC_END", 7)
    expected = nar.synthesize(model, [2, 3], [1] * 11, 42, steps=2, context=15)
    events = []
    rng_state = torch.random.get_rng_state().clone()
    actual = nar.synthesize(model, [2, 3], [1] * 11, 42, steps=2, context=15,
                            on_progress=lambda completed, total: events.append((completed, total)))
    assert events == [(step, 6) for step in range(1, 7)]
    assert torch.equal(actual, expected)
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_synthesis_progress_exception_releases_current_cache(model, monkeypatch):
    monkeypatch.setattr(nar, "CODEC_OFFSET", 8)
    monkeypatch.setattr(nar, "MUSIC_END", 7)
    events, closed = [], []
    original = nar.CachedNAR

    class Tracked(original):
        def close(self):
            super().close()
            closed.append(self)

    def progress(completed, total):
        events.append((completed, total))
        if completed == 3:
            raise RuntimeError("progress handler failed")

    monkeypatch.setattr(nar, "CachedNAR", Tracked)
    with pytest.raises(RuntimeError, match="progress handler failed"):
        nar.synthesize(model, [2, 3], [1] * 11, 42, steps=2, context=15,
                       offload_ar=True, on_progress=progress)
    assert events == [(1, 6), (2, 6), (3, 6)]
    assert len(closed) == 2 and all(not engine.cache for engine in closed)
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())


def test_cancellation_releases_cache(model, monkeypatch):
    monkeypatch.setattr(nar, "CODEC_OFFSET", 8)
    monkeypatch.setattr(nar, "MUSIC_END", 7)
    decisions = iter([False, False, True])
    with patch.object(nar.CachedNAR, "close", autospec=True) as close:
        with pytest.raises(InterruptedError):
            nar.synthesize(model, [2, 3], [1] * 4, 42, cancelled=lambda: next(decisions))
    assert close.call_count == 1


def test_offload_restores_modules_when_generation_raises(monkeypatch):
    class Module:
        def __init__(self):
            self.device, self.moves = torch.device("meta"), []

        def parameters(self):
            return iter([SimpleNamespace(device=self.device)])

        def to(self, *, device):
            self.device = torch.device(device)
            self.moves.append(self.device.type)

    modules = [Module() for _ in range(6)]
    layer = SimpleNamespace(input_layernorm=modules[2], self_attn=modules[3],
                            post_attention_layernorm=modules[4], mlp=modules[5])
    fake = SimpleNamespace(model=SimpleNamespace(embed_tokens=modules[0], layers=[layer]), lm_head=modules[1])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="request failed"):
        with nar._offload_ar(fake, True):
            assert all(module.device.type == "cpu" for module in modules)
            raise RuntimeError("request failed")
    assert all(module.moves == ["cpu", "meta"] for module in modules)
