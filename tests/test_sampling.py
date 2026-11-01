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


def test_compiled_decode_step_is_owned_by_the_model(monkeypatch):
    """Dropping a model must release its compiled decode callable: the cache
    lives on the model, not in a global that strongly references every model."""
    import gc
    import weakref
    from yue2 import sampling

    calls = []
    monkeypatch.setattr(sampling.torch, "compile",
                        lambda fn, dynamic=None: (calls.append(1), fn)[1])

    models = [torch.nn.Linear(2, 2) for _ in range(2)]
    steps = [sampling._ensure_decode_step(m) for m in models]
    assert len(calls) == 2
    # Each model owns exactly one step; a second lookup reuses it.
    assert sampling._ensure_decode_step(models[0]) is steps[0]
    refs = [weakref.ref(m) for m in models]
    # The step is a plain callable: attaching it must not register a child
    # module of the model it wraps (that cycle breaks model.to() traversal).
    for m in models:
        assert "_yue2_decode_step" not in dict(m.named_modules())
        m.to(torch.float64)  # must not recurse
    del models, steps, m
    gc.collect()
    assert all(ref() is None for ref in refs), "model retained by compiled decode cache"
    assert not hasattr(sampling, "_decode_steps")


def test_compiled_decode_step_released_by_pipeline_close(tmp_path, monkeypatch):
    """Pipeline.close() must release the compiled model without waiting for an
    unrelated GC cycle: the decode step holds only a weak model reference."""
    import weakref
    from yue2 import pipeline as pipeline_module
    from yue2 import sampling
    from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM

    monkeypatch.setattr(pipeline_module, "resolve_device", lambda _: torch.device("cpu"))
    monkeypatch.setattr(pipeline_module, "YuE2TextTokenizer", lambda _: None)
    monkeypatch.setattr(pipeline_module, "model_identity", lambda *args, **kwargs: {})
    monkeypatch.setattr(pipeline_module, "empty_cache", lambda device: None)
    cfg = YuE2Config(hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
                     num_key_value_heads=2, head_dim=8, intermediate_size=64,
                     vocab_size=71, max_position_embeddings=64, max_latent_frames=64,
                     pad_token_id=0, eos_token_id=None, bos_token_id=1)
    model = YuE2ForCausalLM(cfg).eval()
    pipe = pipeline_module.YuE2Pipeline(tmp_path, tmp_path, progress=False)
    pipe._model = model
    # Execute the real compiled callable before checking close(), retaining
    # the step to verify that it cannot keep the model alive.
    from yue2.modeling_yue2 import StaticKVCache
    step = sampling._ensure_decode_step(model)
    assert step is model._yue2_decode_step
    assert step.model() is model
    cache = StaticKVCache(cfg.num_hidden_layers, 1, cfg.num_key_value_heads, 64,
                          cfg.head_dim, torch.float32, "cpu")
    with torch.inference_mode():
        step(torch.tensor([[3]]), past_key_values=cache, use_cache=True, logits_to_keep=1)
    ref = weakref.ref(model)
    pipe.close()
    del pipe, model
    # No gc.collect(): a strong back-reference in the step would keep the
    # compiled model (and its device memory) alive until an unrelated cycle.
    assert ref() is None, "close() left the compiled model alive"
    assert step.model() is None
