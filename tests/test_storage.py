import json
from pathlib import Path
import pytest
import threading
from yue2 import storage
from yue2.storage import model_identity, sha256_file
from yue2.storage import collect_hashes


def test_overlapping_writes_use_independent_temporary_files(tmp_path, monkeypatch):
    target = tmp_path / "result.json"
    first_waiting, release = threading.Event(), threading.Event()
    original = storage.os.replace
    errors = []

    def replace(source, destination):
        if threading.current_thread().name == "first-writer":
            first_waiting.set()
            assert release.wait(5)
        return original(source, destination)

    monkeypatch.setattr(storage.os, "replace", replace)

    def first():
        try:
            storage.write_json(target, {"writer": "first"})
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=first, name="first-writer")
    thread.start()
    try:
        assert first_waiting.wait(5)
        storage.write_json(target, {"writer": "second"})
        assert json.loads(target.read_text()) == {"writer": "second"}
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert errors == []
    assert json.loads(target.read_text()) == {"writer": "first"}
    assert list(tmp_path.iterdir()) == [target]


def test_failed_serialization_leaves_previous_receipt_intact(tmp_path):
    target = tmp_path / "result.json"
    storage.write_json(target, {"complete": True})
    with pytest.raises(ValueError):
        storage.write_json(target, {"value": float("nan")})
    assert json.loads(target.read_text()) == {"complete": True}
    assert list(tmp_path.iterdir()) == [target]


def test_replace_failure_removes_temporary_file(tmp_path, monkeypatch):
    target = tmp_path / "result.json"

    def fail(*args):
        raise OSError("synthetic filesystem boundary failure")

    monkeypatch.setattr(storage.os, "replace", fail)
    with pytest.raises(OSError):
        storage.write_json(target, {"complete": True})
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("size", [0, 999, True, "7"])
def test_inconsistent_weight_manifest_size_is_rejected(tmp_path, size):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"fixture")
    (tmp_path / "config.json").write_text("{}")
    manifest = {"files": {weight.name: {"sha256": sha256_file(weight), "bytes": size}}}
    (tmp_path / "weights_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="integrity"):
        model_identity(tmp_path)


def test_matching_size_and_hash_are_accepted(tmp_path):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"fixture")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "weights_manifest.json").write_text(
        json.dumps({"files": {weight.name: {"sha256": sha256_file(weight), "bytes": 7}}})
    )
    assert model_identity(tmp_path)["files"][weight.name]["bytes"] == 7


def test_explicit_verification_opt_out_observes_actual_size(tmp_path):
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"fixture")
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "weights_manifest.json").write_text(
        json.dumps({"files": {weight.name: {"sha256": "bad", "bytes": 999}}})
    )
    assert model_identity(tmp_path, verify=False)["files"][weight.name]["bytes"] == 7


def test_receipt_exclusion_applies_only_to_root_path(tmp_path):
    (tmp_path / "result.json").write_text("{}")
    nested = tmp_path / "stages"
    nested.mkdir()
    (nested / "result.json").write_text("nested evidence")
    (tmp_path / "audio.flac").write_bytes(b"synthetic")
    observed = collect_hashes(tmp_path)
    assert set(observed) == {"stages/result.json", "audio.flac"}
    assert observed["stages/result.json"] == {
        "sha256": sha256_file(nested / "result.json"),
        "bytes": 15,
    }


def test_explicit_relative_exclusion_does_not_hide_same_basename(tmp_path):
    (tmp_path / "stages").mkdir()
    (tmp_path / "other").mkdir()
    (tmp_path / "stages/report.json").write_text("first")
    (tmp_path / "other/report.json").write_text("second")
    assert set(collect_hashes(tmp_path, exclude=("stages/report.json",))) == {"other/report.json"}
