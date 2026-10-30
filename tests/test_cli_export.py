import base64
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from transformers import AutoModelForCausalLM
from yue2.pipeline import YuE2Pipeline
from yue2.modeling_yue2 import YuE2ForCausalLM
from yue2.modeling_vae import YuE2VAE
from test_model import tiny_config as ar_config
from test_vae import tiny_config as vae_config


@pytest.fixture
def sources(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(191)
    ar = YuE2ForCausalLM(ar_config()).eval()
    model = tmp_path / "model"
    vae = tmp_path / "vae"
    ar.save_pretrained(model)
    YuE2VAE(vae_config()).save_pretrained(vae)
    tokens = [bytes([i]) for i in range(256)] + [
        b"\xfffixture" + i.to_bytes(4, "big") for i in range(151643 - 256)
    ]
    (model / "qwen.tiktoken").write_bytes(
        b"".join(
            base64.b64encode(token) + b" " + str(rank).encode() + b"\n"
            for rank, token in enumerate(tokens)
        )
    )
    return model, vae, ar


@torch.inference_mode()
def test_public_export_cli_creates_relocatable_verified_local_pipeline(tmp_path, sources):
    model, vae, ar = sources
    output = tmp_path / "exported"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "yue2.cli",
            "export",
            "--model",
            str(model),
            "--vae",
            str(vae),
            "--device",
            "cpu",
            "--offline",
            "--quiet",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src")),
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["pipeline"] == str(output / "pipeline.json")
    assert record["exported"] is True
    assert (output / "YuE2-3B/qwen.tiktoken").read_bytes() == (model / "qwen.tiktoken").read_bytes()
    restored = AutoModelForCausalLM.from_pretrained(
        output / "YuE2-3B", trust_remote_code=True, local_files_only=True
    ).eval()
    ids = torch.tensor([[3, 5, 8]])
    torch.testing.assert_close(
        restored(ids, use_cache=False).logits, ar(ids, use_cache=False).logits, atol=0, rtol=0
    )
    with YuE2Pipeline.from_pretrained(
        output, device="cpu", local_files_only=True, progress=False
    ) as pipe:
        assert pipe.model_dir == output / "YuE2-3B"
        assert pipe.vae_dir == output / "YuE2-Vae"
        assert pipe.tokenizer.decode(pipe.tokenizer.encode("offline export")) == "offline export"


def test_export_cli_requires_explicit_output():
    result = subprocess.run(
        [sys.executable, "-m", "yue2.cli", "export"],
        capture_output=True,
        text=True,
        env=dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src")),
    )
    assert result.returncode == 2
    assert "required" in result.stderr and "--output" in result.stderr


def test_existing_python_export_api_remains_usable(tmp_path, sources):
    model, vae, _ = sources
    with YuE2Pipeline.from_pretrained(
        model, vae=vae, device="cpu", local_files_only=True, progress=False
    ) as pipe:
        output = tmp_path / "api-export"
        pipe.save_pretrained(output)
        assert (output / "pipeline.json").is_file()
        with pytest.raises(FileExistsError):
            pipe.save_pretrained(output)
