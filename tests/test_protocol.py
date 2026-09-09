import json
from pathlib import Path
import pytest
import torch
from yue2.protocol import *
from yue2.sampling import distribution, window_penalty
from yue2.cli import request_kwargs
from yue2.storage import verify_result, write_json, collect_hashes


class Tokenizer:
    def encode(self, text):
        return [ord(x) for x in text]


def test_defaults_and_native_instructions():
    cfg = GenerationConfig()
    assert cfg.abc == Sampling(.7, .9, 30, 1.005, 100, 32, 4096)
    assert cfg.semantic == Sampling(1., .95, 100, 1.2, 50, 200, 9000)
    assert (cfg.ode_steps, cfg.ode_method, cfg.context) == (32, "midpoint", 24576)
    assert SongRequest("piano", "line\nline").cot == "full"
    assert "chord-annotated" in SongRequest("a", "b").text()
    assert "melody-only" in SongRequest("a", "b", cot="melody").text()
    assert SongRequest("a", "b", cot="off").guidance == 1.01


def test_cfg_keeps_exact_score_and_removes_only_text():
    t = Tokenizer()
    for cot in ("full", "melody"):
        r = SongRequest("STYLE", "LYRIC", cot=cot, cfg_scale=1.2)
        ids = [17, 19, 23]
        positive = token_prefixes(r, t, ids)
        negative = negative_prefix(r, t, ids)
        assert positive[-6:] == negative[-6:] == [ABC_START, 17, 19, 23, ABC_END, MUSIC_START]
        assert negative == [EOD] + t.encode(INSTRUCTIONS[cot]) + negative[-6:]
        assert positive != negative
    r = SongRequest("a", "b", cot="off")
    assert token_prefixes(r, t)[-3:] == [ABC_START, ABC_END, MUSIC_START]
    assert ABC_START not in negative_prefix(r, t)


def test_external_abc_retains_crlf(tmp_path):
    abc = 'X:1\r\nK:C\r\n"C"CDEF|\r\n'
    (tmp_path / "a.abc").write_bytes(abc.encode())
    result = request_kwargs({"id":"test", "tags":"x", "lyrics":"y", "abc_path":"a.abc"}, tmp_path)
    assert result["abc"] == abc
    with pytest.raises(ValueError):
        SongRequest("a", "b", cot="off", abc=abc)
    with pytest.raises(ValueError):
        request_kwargs({"tags":"x", "lyrics":"y", "unknown":1})


def test_window_count_sign_and_window_expiry():
    logits = torch.tensor([[4., -4., 3.]])
    torch.testing.assert_close(window_penalty(logits, [0, 0, 1], 2), torch.tensor([[1., -8., 3.]]))
    torch.testing.assert_close(window_penalty(logits, [2], 2), torch.tensor([[4., -4., 1.5]]))


def test_top_p_off_keeps_three_and_symbolic_one():
    logits = torch.full((1, VOCAB_SIZE), -torch.inf, dtype=torch.bfloat16)
    logits[0, CODEC_OFFSET:CODEC_OFFSET+5] = torch.tensor([10., 1., 0., -1., -2.])
    config = Sampling(1, .5, 5, 1, 50, 0, 5)
    off = distribution(logits, config, [], 0, "semantic", legacy_off=True)
    symbolic = distribution(logits, config, [], 0, "semantic", legacy_off=False)
    assert torch.isfinite(off).sum() == 3
    assert torch.isfinite(symbolic).sum() == 1
    assert off.dtype == torch.bfloat16 and symbolic.dtype == torch.float32


def test_vocab_and_eos_minimum():
    logits = torch.zeros(1, VOCAB_SIZE)
    config = Sampling(0, 1., 100, 1, 50, 2, 5)
    score = distribution(logits, config, [], 0, "semantic")
    assert not torch.isfinite(score[0, MUSIC_END])
    assert not torch.isfinite(score[0, EOD])
    assert torch.isfinite(score[0, CODEC_OFFSET:CODEC_OFFSET+CODEC_SIZE]).all()
    assert torch.isfinite(distribution(logits, config, [], 2, "semantic")[0, MUSIC_END])


def test_integrity_rejects_incomplete_outputs(tmp_path):
    write_json(tmp_path / "result.json", {"status":"complete", "identity":"x", "artifacts":{}})
    with pytest.raises(ValueError, match="Incomplete"):
        verify_result(tmp_path, "x")


def test_invalid_hyperparameters():
    for value in (float("nan"), float("inf"), -1, 21):
        with pytest.raises(ValueError):
            SongRequest("x", "y", cfg_scale=value)
    with pytest.raises(ValueError):
        Sampling(min_tokens=20, max_tokens=10)
    with pytest.raises(ValueError):
        GenerationConfig(context=1024)


def test_partial_abc_override_keeps_abc_defaults():
    config = GenerationConfig.from_dict({"abc": {"temperature": .4}})
    assert config.abc == Sampling(.4, .9, 30, 1.005, 100, 32, 4096)
    assert resolve_sampling({"top_k": 12}, GenerationConfig().abc).max_tokens == 4096


def test_saved_plan_preserves_exact_ids_and_rejects_edits(tmp_path):
    from yue2 import SymbolicPlan
    request = SongRequest("style", "lyrics", abc="ABC")
    plan = SymbolicPlan(request, "ABC", [65,66,67], token_prefixes(request, Tokenizer(), [65,66,67]))
    plan.save(tmp_path)
    assert SymbolicPlan.load(tmp_path) == plan
    (tmp_path / "score.abc").write_text("DEF")
    with pytest.raises(ValueError, match="changed"):
        SymbolicPlan.load(tmp_path)
    with pytest.raises(ValueError, match="ordinary"):
        token_prefixes(request, Tokenizer(), [ABC_END])
