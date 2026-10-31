import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import soundfile as sf
from yue2.pipeline import SymbolicPlan, SemanticResult, SongResult
from yue2.protocol import SongRequest
from yue2.storage import verify_result


@pytest.fixture
def saved(tmp_path):
    request = SongRequest(
        style="piano",
        lyrics="[Verse]\nHello",
        cot="melody",
        abc="X:1\r\nK:C\r\nC D|\r\n",
    )
    plan = SymbolicPlan(
        request, request.abc, [70, 71], [1, 70, 71], {"seconds": 0.1}, True
    )
    semantic = SemanticResult(plan, [3, 4], {"seconds": 0.2}, False)
    audio = (
        np.stack(
            [np.sin(np.arange(4800) * 0.03), np.cos(np.arange(4800) * 0.03)], axis=-1
        )
        * 0.1
    )
    latents = np.arange(128, dtype=np.float32).reshape(1, 64, 2) / 128
    song = SongResult(
        audio,
        48000,
        semantic,
        latents,
        {"native": "fixture"},
        {"mot": {}, "vae": {}},
        {"semantic": semantic.timing, "e2e_seconds": 0.3},
        "fixture-id",
    )
    directory = tmp_path / "native"
    song.save_artifacts(directory)
    return directory, song


def test_load_restores_exact_stored_run_and_can_export_audio(saved, tmp_path):
    directory, original = saved
    loaded = SongResult.load(directory, expected_identity="fixture-id")
    stored, rate = sf.read(directory / "audio.flac", dtype="float32", always_2d=True)
    np.testing.assert_array_equal(loaded.audio, stored)
    assert loaded.sample_rate == rate == 48000
    np.testing.assert_array_equal(loaded.latents, original.latents)
    assert loaded.semantic.tokens == original.semantic.tokens
    assert loaded.semantic.plan == original.semantic.plan
    assert loaded.config == original.config and loaded.weights == original.weights
    assert (
        loaded.timing == original.timing
        and loaded.request_identity == original.request_identity
    )
    assert loaded.truncated == {"abc": True, "semantic": False}
    output = tmp_path / "restored.wav"
    loaded.save(output)
    np.testing.assert_array_equal(sf.read(output, dtype="float32")[0], stored)
    copied = tmp_path / "copy"
    loaded.save_artifacts(copied)
    assert verify_result(copied)["identity"] == "fixture-id"
    assert (copied / "score.abc").read_bytes() == (directory / "score.abc").read_bytes()


@pytest.mark.parametrize(
    "artifact",
    ["audio.flac", "semantic.npy", "latent.npy", "request.json", "config.json"],
)
def test_load_refuses_changed_native_artifacts(saved, artifact):
    directory, _ = saved
    with (directory / artifact).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="corrupt result"):
        SongResult.load(directory)


def test_load_refuses_a_different_request_identity(saved):
    directory, _ = saved
    with pytest.raises(ValueError, match="identity changed"):
        SongResult.load(directory, expected_identity="different-request")


def test_load_does_not_import_generation_models(saved, tmp_path):
    directory, _ = saved
    boundary = tmp_path / "no-models"
    boundary.mkdir()
    (boundary / "sitecustomize.py").write_text(
        "import sys\nclass Boundary:\n    def find_spec(self, fullname, path=None, target=None):\n        if fullname in {'yue2.modeling_yue2','yue2.modeling_vae'}: raise RuntimeError('result loading imported a generation model')\nsys.meta_path.insert(0,Boundary())\n"
    )
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            (str(boundary), str(Path(__file__).resolve().parents[1] / "src"))
        ),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from yue2 import SongResult; import sys; s=SongResult.load(sys.argv[1]); assert s.request_identity=='fixture-id'",
            str(directory),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_existing_plan_and_receipt_loading_control(saved):
    directory, song = saved
    assert SymbolicPlan.load(directory) == song.semantic.plan
    assert verify_result(directory)["identity"] == song.request_identity
