"""Explicit generation state machine: run_state.json records exactly which lifecycle state a
run reached (or failed at), rejects transitions and manifests that don't make sense, and lets
generate_resumable() recover from a corrupted/inconsistent manifest instead of crashing."""
import json

import pytest

from yue2 import cli
from yue2.pipeline import RunState, YuE2Pipeline
from test_resume import stub_pipe, zero_calls


def write_run_state(directory, **fields):
    directory.mkdir(parents=True, exist_ok=True)
    base = {"identity": "x", "stage_hashes": {}, "state": "created", "schema_version": 2}
    (directory / "run_state.json").write_text(json.dumps({**base, **fields}))


def test_full_run_ends_in_complete_with_both_stage_hashes_recorded(tmp_path):
    calls = zero_calls()
    directory = tmp_path / "out"
    stub_pipe(tmp_path, calls).generate_resumable(directory, style="piano", lyrics="la la", id="s")
    state = RunState.load(directory)
    assert state.state == "complete"
    assert set(state.stage_hashes) == {"semantic", "synthesis"}


@pytest.mark.parametrize("fail_at,expected_state", [
    ("plan", "plan_failed"),
    ("semantic", "semantic_failed"),
    ("synthesis", "synthesis_failed"),
    ("decode", "decode_failed"),
])
def test_a_crash_records_the_matching_failed_state(tmp_path, fail_at, expected_state):
    calls = zero_calls()
    directory = tmp_path / "out"
    with pytest.raises(RuntimeError):
        stub_pipe(tmp_path, calls, fail_at=fail_at).generate_resumable(
            directory, style="piano", lyrics="la la", id="s")
    assert RunState.load(directory).state == expected_state


def test_transition_rejects_a_jump_that_skips_a_stage(tmp_path):
    state = RunState("id")
    state.transition(tmp_path, "planned")
    with pytest.raises(ValueError, match="Invalid generation state transition"):
        state.transition(tmp_path, "synthesized")  # skips semantic_generated
    assert state.state == "planned"  # the rejected attempt did not partially apply


def test_transition_allows_a_retry_that_fails_again(tmp_path):
    state = RunState("id")
    state.transition(tmp_path, "planned")
    state.transition(tmp_path, "semantic_failed")
    state.transition(tmp_path, "semantic_failed")
    assert state.state == "semantic_failed"


def test_rebase_bypasses_validation_for_a_deliberate_downgrade(tmp_path):
    state = RunState("id")
    state.transition(tmp_path, "planned")
    state.transition(tmp_path, "semantic_generated", {"semantic": {"a.npy": {"sha256": "x", "bytes": 1}}})
    state.transition(tmp_path, "synthesized", {"synthesis": {"b.npy": {"sha256": "y", "bytes": 2}}})
    with pytest.raises(ValueError, match="Invalid generation state transition"):
        state.transition(tmp_path, "planned")

    state.rebase(tmp_path, "planned")
    assert state.state == "planned"
    assert RunState.load(tmp_path).state == "planned"


@pytest.mark.parametrize("bad_fields", [
    {"state": "synthesized", "stage_hashes": {}},  # claims progress no hash backs up
    {"state": "not_a_real_state"},
    {"schema_version": 1},
])
def test_load_rejects_an_inconsistent_or_outdated_manifest(tmp_path, bad_fields):
    write_run_state(tmp_path, **bad_fields)
    assert RunState.load(tmp_path) is None


def test_cli_failure_json_reports_the_state_the_run_stopped_at(tmp_path, monkeypatch):
    calls = zero_calls()
    monkeypatch.setenv("YUE2_KIT", str(tmp_path))
    monkeypatch.setattr(YuE2Pipeline, "from_pretrained",
                        lambda *a, **k: stub_pipe(tmp_path, calls, fail_at="synthesis"))
    output = tmp_path / "runs"
    argv = ["generate", "--output", str(output), "--id", "s", "--style", "piano", "--lyrics", "la la"]
    with pytest.raises(RuntimeError):
        cli.main(argv)
    failure = json.loads((output / "s" / "failure.json").read_text())
    assert failure["state"] == "synthesis_failed"


def test_generate_resumable_recovers_from_an_inconsistent_manifest_by_redoing_everything(tmp_path):
    calls = zero_calls()
    directory = tmp_path / "out"
    write_run_state(directory, identity="bogus", state="synthesized", stage_hashes={})

    receipt = stub_pipe(tmp_path, calls).generate_resumable(
        directory, style="piano", lyrics="la la", id="s", resume=True)
    assert receipt["resumed"] is False
    assert calls == {"plan": 1, "semantic": 1, "synthesis": 1, "decode": 1}
    assert RunState.load(directory).state == "complete"
