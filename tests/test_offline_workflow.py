"""Exercise the public workflow with real, small models and no downloads."""

import base64
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf
import torch

from yue2.modeling_vae import YuE2VAE, YuE2VAEConfig
from yue2.modeling_yue2 import YuE2Config, YuE2ForCausalLM
from yue2.pipeline import SongResult, YuE2Pipeline
from yue2.protocol import GenerationConfig, Sampling
from yue2.storage import sha256_file


@pytest.fixture(scope="module")
def offline_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("offline-models")
    torch.set_num_threads(1)
    torch.manual_seed(819)
    model_dir, vae_dir = root / "model", root / "vae"
    # Keep the full native token vocabulary and VAE strides; reduce model width.
    config = YuE2Config(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        intermediate_size=16,
        max_latent_frames=32,
    )
    model = YuE2ForCausalLM(config).eval()
    # Tied logits select the first allowed codec token, then EOS at min_tokens.
    with torch.no_grad():
        model.lm_head.weight.zero_()
    model.save_pretrained(model_dir)
    common = dict(channels=1, c_mults=[1] * 6, strides=[2, 2, 4, 4, 5, 6], use_snake=True)
    vae = YuE2VAE(
        YuE2VAEConfig(
            encoder_config=dict(common, in_channels=2, latent_dim=128),
            decoder_config=dict(
                common, out_channels=2, latent_dim=64, snake_type="vanilla", final_tanh=False
            ),
            latent_dim=64,
        )
    )
    vae.eval().save_pretrained(vae_dir)
    tokens = [bytes([i]) for i in range(256)]
    tokens += [b"\xfffixture" + i.to_bytes(4, "big") for i in range(151643 - 256)]
    (model_dir / "qwen.tiktoken").write_bytes(
        b"".join(
            base64.b64encode(token) + b" " + str(rank).encode() + b"\n"
            for rank, token in enumerate(tokens)
        )
    )
    generation = GenerationConfig(
        semantic=Sampling(temperature=0, min_tokens=3, max_tokens=4), ode_steps=2
    )
    settings = root / "generation.json"
    settings.write_text(json.dumps(generation.to_dict()))
    return model_dir, vae_dir, generation, settings


def cli(*args, returncode=0):
    env = dict(
        os.environ,
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
    )
    result = subprocess.run(
        [sys.executable, "-m", "yue2.cli", *map(str, args)],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == returncode, result.stderr
    return json.loads(result.stdout) if result.stdout.strip() else None


def open_pipeline(models):
    model, vae, generation, _ = models
    return YuE2Pipeline.from_pretrained(
        model,
        vae=vae,
        generation_config=generation,
        device="cpu",
        backend="torch-eager",
        vae_core_frames=2,
        progress=False,
        local_files_only=True,
    )


def test_real_generation_callback_retry_save_load_and_decode(offline_models, tmp_path):
    request = dict(
        style="piano", lyrics="", cot="full", seed=17, abc="X:1\r\nM:4/4\r\nK:C\r\nCDEF|\r\n"
    )
    events = []
    with open_pipeline(offline_models) as pipe:

        def cancelled(stage, completed, total):
            if stage == "nar":
                raise InterruptedError("caller stopped synthesis")

        with pytest.raises(InterruptedError, match="caller stopped"):
            pipe(**request, on_progress=cancelled)
        song = pipe(**request, on_progress=lambda *event: events.append(event))
        repeat = pipe(**request)
    np.testing.assert_array_equal(song.latents, repeat.latents)
    np.testing.assert_array_equal(song.audio, repeat.audio)
    assert events == [("nar", 1, 2), ("nar", 2, 2), ("vae", 1, 2), ("vae", 2, 2)]
    directory = tmp_path / "saved"
    song.save_artifacts(directory)
    restored = SongResult.load(directory, expected_identity=song.request_identity)
    assert (directory / "score.abc").read_bytes() == request["abc"].encode()
    assert restored.semantic.tokens == song.semantic.tokens
    assert restored.config == song.config and restored.weights == song.weights
    assert restored.truncated == song.truncated
    assert song.truncated == {"abc": False, "semantic": False}
    np.testing.assert_array_equal(restored.latents, song.latents)
    np.testing.assert_allclose(restored.audio, song.audio, atol=2**-23, rtol=0)
    report = cli("inspect", directory)
    assert report["artifacts_verified"] and report["identity"] == song.request_identity
    output = tmp_path / "decoded.wav"
    cli(
        "decode",
        directory / "latent.npy",
        "--vae",
        offline_models[1],
        "--offline",
        "--vae-core-frames",
        "2",
        "--output",
        output,
    )
    audio, rate = sf.read(output, dtype="float32")
    np.testing.assert_array_equal(audio, song.audio)
    assert rate == song.sample_rate


def test_real_batch_preflight_resume_and_relocated_export(offline_models, tmp_path):
    model, vae, _, settings = offline_models
    request = dict(
        id="instrumental",
        style="piano",
        lyrics="",
        cot="off",
        seed=19,
        semantic_sampling={"max_tokens": 3},
    )
    source = tmp_path / "batch.jsonl"
    source.write_text("\n".join(map(json.dumps, [dict(request, id="invalid", seed=-1), request])))
    flags = [
        "--model",
        model,
        "--vae",
        vae,
        "--config",
        settings,
        "--device",
        "cpu",
        "--backend",
        "torch-eager",
        "--offline",
        "--quiet",
    ]
    report = cli("validate", "--input", source, "--config", settings, returncode=1)
    assert [row["ok"] for row in report["results"]] == [False, True]
    runs = tmp_path / "runs"
    cli("batch", "--input", source, "--output", runs, *flags, returncode=1)
    receipt = json.loads((runs / "batch.json").read_text())
    assert receipt["failed"] == 1 and [row["status"] for row in receipt["results"]] == [
        "failed",
        "complete",
    ]
    directory = runs / "instrumental"
    song = SongResult.load(directory)
    assert song.truncated == {"abc": False, "semantic": True}
    before = {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()}
    source.write_text(json.dumps(request) + "\n")
    cli("batch", "--input", source, "--output", runs, "--resume", *flags)
    assert before == {p.name: sha256_file(p) for p in directory.iterdir() if p.is_file()}
    exported = tmp_path / "exported"
    cli("export", "--output", exported, *flags)
    relocated = tmp_path / "relocated"
    exported.rename(relocated)
    # The saved pipeline supplies model/VAE paths and generation defaults offline.
    request_file = tmp_path / "song.json"
    request_file.write_text(json.dumps(request))
    cli(
        "generate",
        "--request",
        request_file,
        "--model",
        relocated,
        "--offline",
        "--device",
        "cpu",
        "--backend",
        "torch-eager",
        "--quiet",
        "--output",
        tmp_path / "replayed",
    )
    replayed = SongResult.load(tmp_path / "replayed" / "instrumental")
    assert replayed.semantic.tokens == song.semantic.tokens
    np.testing.assert_array_equal(replayed.latents, song.latents)
    np.testing.assert_array_equal(replayed.audio, song.audio)


def test_instrumental_preview_preserves_native_and_reports_short_source(
    offline_models, tmp_path, monkeypatch, capsys
):
    root = Path(__file__).resolve().parents[1]
    helper = root / "skills/yue2-music/instrumental/scripts/instrumental.py"
    prepared = tmp_path / "prepared"
    subprocess.run(
        [
            sys.executable,
            "-B",
            str(helper),
            "prepare",
            "--abc",
            str(helper.parents[1] / "assets/example-ins.abc"),
            "--output",
            str(prepared),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    script = root / "examples/instrumental_clip.py"
    spec = importlib.util.spec_from_file_location("instrumental_clip", script)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    # Redirect only the checkpoint location; execute the real model and file APIs.
    pipe = open_pipeline(offline_models)
    monkeypatch.setattr(YuE2Pipeline, "from_pretrained", lambda **kwargs: pipe)
    for seconds, expected_code in [(0.05, 0), (1, 1)]:
        output = tmp_path / f"preview-{seconds}"
        monkeypatch.setattr(
            sys,
            "argv",
            [
                str(script),
                "--prepared",
                str(prepared),
                "--seconds",
                str(seconds),
                "--device",
                "cpu",
                "--output",
                str(output),
            ],
        )
        assert example.main() == expected_code
        record = json.loads(capsys.readouterr().out)
        native = SongResult.load(output / "native")
        clip, rate = sf.read(output / "clip.wav", dtype="float32")
        assert len(clip) == min(round(seconds * rate), len(native.audio))
        np.testing.assert_allclose(clip, native.audio[: len(clip)], atol=2**-23, rtol=0)
        assert record["shorter_than_requested"] is bool(expected_code)
        assert record["generation_truncated"] == {"abc": False, "semantic": False}
