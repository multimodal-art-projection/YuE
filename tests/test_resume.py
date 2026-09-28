"""Stage-level checkpointing: an interrupted generate_resumable() run continues from the
last verified stage on disk instead of recomputing already-finished work."""
import json

import numpy as np
import pytest

from yue2 import cli
from yue2.pipeline import YuE2Pipeline, SymbolicPlan, SemanticResult
from yue2.protocol import GenerationConfig


def stub_pipe(tmp_path, calls, fail_at=None):
    """A YuE2Pipeline with plan/generate_semantic/synthesize/decode replaced by counting fakes,
    so stage-checkpointing behavior can be tested without GPU weights."""
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir(exist_ok=True)
    (vae_dir / "config.json").write_text(json.dumps({"release_variant": "test"}))

    pipe = object.__new__(YuE2Pipeline)
    pipe.progress = False
    pipe.weights = {"mot": "mot-hash", "vae": "vae-hash"}
    pipe.runtime_sha256 = "r" * 64
    pipe.generation_config = GenerationConfig()
    pipe.backend, pipe.quantization = "torch", "none"
    pipe.vae_core_frames, pipe.offload_ar = 1024, False
    pipe.device, pipe.memory_budget_gib = "cpu", 24
    pipe.load_timing = {}
    pipe.vae_dir = vae_dir

    def fake_plan(*, request, abc_sampling=None, cancelled=None, on_token=None):
        calls["plan"] += 1
        if fail_at == "plan":
            raise RuntimeError("boom-plan")
        ids = [10, 11, 12]
        return SymbolicPlan(request, "X:1\nK:C\nCDEF|", ids, [0] + ids, {"seconds": 0.1}, False)

    def fake_generate_semantic(plan, *, sampling=None, cancelled=None, on_token=None):
        calls["semantic"] += 1
        if fail_at == "semantic":
            raise RuntimeError("boom-semantic")
        return SemanticResult(plan, [100, 101, 102], {"seconds": 0.2}, False)

    def fake_synthesize(semantic, *, cancelled=None):
        calls["synthesis"] += 1
        if fail_at == "synthesis":
            raise RuntimeError("boom-synthesis")
        return np.arange(4 * 64, dtype=np.float32).reshape(4, 64)

    def fake_decode(latents, *, full=False, vae=None):
        calls["decode"] += 1
        if fail_at == "decode":
            raise RuntimeError("boom-decode")
        return np.zeros((10, 2), dtype=np.float32)

    pipe.plan = fake_plan
    pipe.generate_semantic = fake_generate_semantic
    pipe.synthesize = fake_synthesize
    pipe.decode = fake_decode
    pipe.close = lambda: None
    return pipe


def zero_calls():
    return {"plan": 0, "semantic": 0, "synthesis": 0, "decode": 0}


def test_fresh_run_persists_every_stage_checkpoint(tmp_path):
    calls = zero_calls()
    directory = tmp_path / "out"
    receipt = stub_pipe(tmp_path, calls).generate_resumable(directory, style="piano", lyrics="la la", id="s")
    assert receipt["resumed"] is False
    assert calls == {"plan": 1, "semantic": 1, "synthesis": 1, "decode": 1}
    for name in ("plan.json", "plan_manifest.json", "semantic.npy", "semantic_meta.json",
                "latent.npy", "synthesis_meta.json", "run_state.json", "result.json"):
        assert (directory / name).exists(), name


def test_resume_with_no_crash_reuses_the_completed_result(tmp_path):
    calls = zero_calls()
    directory = tmp_path / "out"
    kwargs = dict(style="piano", lyrics="la la", id="s")
    stub_pipe(tmp_path, calls).generate_resumable(directory, **kwargs)
    assert calls == {"plan": 1, "semantic": 1, "synthesis": 1, "decode": 1}
    receipt = stub_pipe(tmp_path, calls).generate_resumable(directory, resume=True, **kwargs)
    assert receipt["resumed"] is True
    # Nothing was recomputed; the existing result.json was verified and returned directly.
    assert calls == {"plan": 1, "semantic": 1, "synthesis": 1, "decode": 1}


@pytest.mark.parametrize("fail_at,before_crash,after_resume", [
    # A stage that had already checkpointed to disk before the crash is reused on resume
    # (its count does not increase again); the failed stage and everything after it reruns.
    ("semantic", {"plan": 1, "semantic": 1, "synthesis": 0, "decode": 0},
                {"plan": 1, "semantic": 2, "synthesis": 1, "decode": 1}),
    ("synthesis", {"plan": 1, "semantic": 1, "synthesis": 1, "decode": 0},
                 {"plan": 1, "semantic": 1, "synthesis": 2, "decode": 1}),
    ("decode", {"plan": 1, "semantic": 1, "synthesis": 1, "decode": 1},
              {"plan": 1, "semantic": 1, "synthesis": 1, "decode": 2}),
])
def test_crash_resumes_from_the_last_verified_stage(tmp_path, fail_at, before_crash, after_resume):
    calls = zero_calls()
    directory = tmp_path / "out"
    kwargs = dict(style="piano", lyrics="la la", id="s")
    with pytest.raises(RuntimeError, match=f"boom-{fail_at}"):
        stub_pipe(tmp_path, calls, fail_at=fail_at).generate_resumable(directory, **kwargs)
    assert calls == before_crash
    assert not (directory / "result.json").exists()

    receipt = stub_pipe(tmp_path, calls).generate_resumable(directory, resume=True, **kwargs)
    assert receipt["resumed"] is False
    assert (directory / "result.json").exists()
    assert calls == after_resume


def test_resume_without_the_flag_refuses_a_nonempty_directory(tmp_path):
    calls = zero_calls()
    directory = tmp_path / "out"
    kwargs = dict(style="piano", lyrics="la la", id="s")
    with pytest.raises(RuntimeError, match="boom-synthesis"):
        stub_pipe(tmp_path, calls, fail_at="synthesis").generate_resumable(directory, **kwargs)
    with pytest.raises(FileExistsError):
        stub_pipe(tmp_path, calls).generate_resumable(directory, **kwargs)


def test_changed_request_on_resume_is_rejected_not_silently_mixed(tmp_path):
    calls = zero_calls()
    directory = tmp_path / "out"
    with pytest.raises(RuntimeError, match="boom-synthesis"):
        stub_pipe(tmp_path, calls, fail_at="synthesis").generate_resumable(
            directory, style="piano", lyrics="la la", id="s")
    with pytest.raises(ValueError, match="identity changed"):
        stub_pipe(tmp_path, calls).generate_resumable(
            directory, style="piano", lyrics="a totally different song", id="s", resume=True)


def test_tampered_semantic_artifact_forces_semantic_and_synthesis_to_redo(tmp_path):
    calls = zero_calls()
    directory = tmp_path / "out"
    kwargs = dict(style="piano", lyrics="la la", id="s")
    with pytest.raises(RuntimeError, match="boom-decode"):
        stub_pipe(tmp_path, calls, fail_at="decode").generate_resumable(directory, **kwargs)
    assert calls == {"plan": 1, "semantic": 1, "synthesis": 1, "decode": 1}
    (directory / "semantic.npy").write_bytes(b"corrupted-bytes")

    receipt = stub_pipe(tmp_path, calls).generate_resumable(directory, resume=True, **kwargs)
    assert receipt["resumed"] is False
    # Plan was untouched and still verifies; semantic and its dependent synthesis stage
    # cannot be trusted once the semantic artifact fails its hash check, so both redo.
    assert calls == {"plan": 1, "semantic": 2, "synthesis": 2, "decode": 2}


def test_missing_plan_forces_every_later_stage_to_redo_even_if_still_hash_valid(tmp_path):
    calls = zero_calls()
    directory = tmp_path / "out"
    kwargs = dict(style="piano", lyrics="la la", id="s")
    with pytest.raises(RuntimeError, match="boom-decode"):
        stub_pipe(tmp_path, calls, fail_at="decode").generate_resumable(directory, **kwargs)
    assert calls == {"plan": 1, "semantic": 1, "synthesis": 1, "decode": 1}
    (directory / "plan_manifest.json").unlink()

    receipt = stub_pipe(tmp_path, calls).generate_resumable(directory, resume=True, **kwargs)
    assert receipt["resumed"] is False
    # A regenerated plan can never be paired with a semantic/synthesis stage produced by the
    # previous (now-lost) plan, even though those artifacts still hash-verify on their own.
    assert calls == {"plan": 2, "semantic": 2, "synthesis": 2, "decode": 2}


def test_cli_generate_resume_continues_after_a_crash(tmp_path, monkeypatch):
    calls = zero_calls()
    monkeypatch.setenv("YUE2_KIT", str(tmp_path))
    monkeypatch.setattr(YuE2Pipeline, "from_pretrained",
                        lambda *a, **k: stub_pipe(tmp_path, calls, fail_at="synthesis"))
    output = tmp_path / "runs"
    argv = ["generate", "--output", str(output), "--id", "s", "--style", "piano", "--lyrics", "la la"]
    with pytest.raises(RuntimeError, match="boom-synthesis"):
        cli.main(argv)
    assert calls == {"plan": 1, "semantic": 1, "synthesis": 1, "decode": 0}
    assert (output / "s" / "failure.json").exists()
    assert not (output / "s" / "result.json").exists()

    monkeypatch.setattr(YuE2Pipeline, "from_pretrained", lambda *a, **k: stub_pipe(tmp_path, calls))
    assert cli.main(argv + ["--resume"]) == 0
    assert calls == {"plan": 1, "semantic": 1, "synthesis": 2, "decode": 1}
    assert (output / "s" / "result.json").exists()
