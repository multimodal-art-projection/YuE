"""Small CLI sharing the Python pipeline's defaults and artifact protocol."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from .storage import write_json, verify_result, identity
from .protocol import GenerationConfig, SongRequest, preflight


def kit_root():
    return Path(os.environ.get("YUE2_KIT", Path(__file__).resolve().parents[2]))


def model_paths(args):
    root = kit_root()
    local_model = root / "models/YuE2-3B"
    local_vae = root / "models" / ("YuE2-Vae-legacy" if args.vae == "legacy" else "YuE2-Vae")
    model = args.model or (str(local_model) if local_model.exists() else "m-a-p/YuE2-3B")
    vae = args.vae if args.vae not in {"standard", "legacy"} else (
        str(local_vae) if local_vae.exists() else "m-a-p/" + local_vae.name)
    return model, vae


def get_pipe(args):
    from .pipeline import YuE2Pipeline
    model, vae = model_paths(args)
    config = generation_config(args)
    return YuE2Pipeline.from_pretrained(model, vae=vae, revision=args.revision,
             vae_revision=args.vae_revision, device=args.device, memory_budget_gib=args.budget,
             backend=args.backend, quantization=args.quantization, offload_ar=args.offload_ar,
             local_files_only=args.offline, generation_config=config,
             vae_core_frames=512 if args.budget <= 12 else 1024,
             progress=not getattr(args, "quiet", False))


def generation_config(args):
    if args.config:
        data = json.loads(Path(args.config).read_text(encoding="utf-8"))
    elif getattr(args, "model", None) and (Path(args.model) / "pipeline.json").is_file():
        data = json.loads((Path(args.model) / "pipeline.json").read_text(encoding="utf-8"))["generation_config"]
    else:
        return GenerationConfig()
    return GenerationConfig.from_dict(data)


def batch_ids(rows):
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Batch input must contain request objects")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(x, str) for x in ids) or len(ids) != len(set(ids)):
        raise ValueError("Every batch request must have a unique string id")
    for identifier in ids:
        SongRequest(style="", lyrics="", id=identifier)


def validate_requests(args):
    path = Path(args.request or args.input)
    rows = ([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if args.input else [json.loads(path.read_text(encoding="utf-8"))])
    if args.input:
        batch_ids(rows)
    config, results = generation_config(args), []
    for row in rows:
        try:
            request = preflight(request_kwargs(row, path.parent), config)
            results.append({"id": request.id, "ok": True})
        except (ValueError, TypeError, OSError, KeyError) as exc:
            results.append({"ok": False, "error": str(exc), "type": type(exc).__name__})
    report = {"ok": all(row["ok"] for row in results), "checked": len(results), "results": results}
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


def request_kwargs(data, base=None):
    if not isinstance(data, dict):
        raise ValueError("Request must be a JSON object")
    base = Path.cwd() if base is None else base
    allowed = {"style", "tags", "lyrics", "cot", "seed", "abc", "cfg_scale", "id", "abc_sampling", "semantic_sampling"}
    metadata = {"lang", "eval_index", "clip_id", "prompt"}
    unknown = set(data) - allowed - metadata - {"abc_path"}
    if unknown:
        raise ValueError(f"Unknown request fields: {sorted(unknown)}")
    result = {k: v for k, v in data.items() if k in allowed}
    if "abc_path" in data:
        if data.get("abc") is not None:
            raise ValueError("Pass abc or abc_path, not both")
        result["abc"] = (base / data["abc_path"]).read_bytes().decode("utf-8")
    if data.get("prompt") is not None:
        from .protocol import SongRequest
        request = SongRequest(style=result.get("style", result.get("tags")), lyrics=result["lyrics"],
                              cot=result.get("cot", "full"))
        if data["prompt"] != request.text():
            raise ValueError("Historical literal prompt does not match native instruction/style/lyrics")
    return result


def export_models(args):
    """Export a verified, portable local pipeline without generating a song."""
    directory = Path(args.output)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Export needs an empty output directory")
    pipe = get_pipe(args)
    try:
        pipe.save_pretrained(directory)
    finally:
        pipe.close()
    print(json.dumps({"exported": True, "pipeline": str(directory / "pipeline.json")}))
    return 0


def inspect_run(args):
    """Verify a saved run without loading generation models."""
    result = verify_result(args.directory)
    report = {"artifacts_verified": True, "identity": result["identity"],
              "sample_rate": result["sample_rate"], "audio_seconds": result["audio_seconds"],
              "truncated": result["truncated"],
              "needs_review": any(result["truncated"].values()), "quality_checked": False}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def decode_latents(args):
    """Decode a standalone latent array with only the FP32 VAE loaded."""
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if output.suffix.lower() not in {".flac", ".wav"}:
        raise ValueError("Use .flac or .wav")
    if args.vae_core_frames < 1:
        raise ValueError("vae-core-frames must be positive")
    import numpy as np
    import soundfile as sf
    with open(args.latent, "rb") as stream:
        latent = np.load(stream, allow_pickle=False)
        if not isinstance(latent, np.ndarray):
            latent.close()
            raise ValueError("Expected a NumPy .npy array")
    if latent.dtype.kind != "f" or not np.isfinite(latent).all():
        raise ValueError("Latents must contain finite floating-point values")
    if (latent.ndim not in (2, 3) or not latent.size
            or (latent.ndim == 3 and latent.shape[0] != 1)):
        raise ValueError("Expected nonempty [T,C] or [1,C,T] latents")
    from .modeling_vae import YuE2VAE, YuE2VAEConfig
    from .storage import resolve_model, model_identity
    source = resolve_model(args.vae, revision=args.vae_revision, local_files_only=args.offline)
    config = YuE2VAEConfig.from_pretrained(source, local_files_only=True)
    if latent.ndim == 2 and latent.shape[1] == config.latent_dim:
        latent = latent.T[None]
    if latent.ndim != 3 or latent.shape[:2] != (1, config.latent_dim) or latent.shape[-1] < 1:
        raise ValueError(f"Expected nonempty [T,{config.latent_dim}] or [1,{config.latent_dim},T] latents")
    weights = model_identity(source)
    import torch
    device = torch.device(args.device)
    model = YuE2VAE.from_pretrained(source, config=config, decoder_only=True,
                                    device=device, local_files_only=True)
    try:
        audio = model.decode_tiled(latent, core_frames=args.vae_core_frames,
                                   halo_frames=16, output_device="cpu")[0].transpose(0, 1).numpy()
        if not np.isfinite(audio).all():
            raise ValueError("VAE produced non-finite audio")
        audio = audio.clip(-1, 1)
    finally:
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        sf.write(stream, audio, config.sample_rate,
                 format="FLAC" if output.suffix.lower() == ".flac" else "WAV",
                 subtype="PCM_24" if output.suffix.lower() == ".flac" else "FLOAT")
    print(json.dumps({"operation": "decode_latents", "output": str(output),
                      "decoder_only": True, "sample_rate": config.sample_rate,
                      "audio_seconds": len(audio) / config.sample_rate,
                      "weights": weights, "quality_checked": False}))
    return 0


def doctor(args):
    import torch
    from .storage import model_identity
    model, vae = model_paths(args)
    versions = {}
    for package in ("torch", "transformers", "huggingface-hub", "safetensors", "tiktoken", "soundfile"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    devices = []
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        devices.append({"id": i, "name": p.name, "memory_gib": p.total_memory / 2**30,
                        "compute_capability": [p.major, p.minor]})
    report = {"dependencies_ready": all(versions.values()), "versions": versions,
              "cuda": devices, "mps_available": torch.backends.mps.is_available(),
              "model": model, "vae": vae, "default_cot": "full", "default_cfg": {"full": 1., "melody": 1., "off": 1.01},
              "validated": False, "note": "Environment readiness is not quality or real-24GB acceptance."}
    if args.verify_hashes:
        from .storage import resolve_model
        report["weights"] = {"model": model_identity(resolve_model(model, revision=args.revision, local_files_only=args.offline)),
                             "vae": model_identity(resolve_model(vae, revision=args.vae_revision, local_files_only=args.offline))}
    if args.output:
        write_json(args.output, report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["dependencies_ready"] else 1


def generate(args):
    if args.request:
        path = Path(args.request)
        data = json.loads(path.read_text(encoding="utf-8"))
        base = path.parent
    else:
        data, base = {}, Path.cwd()
    for key in ("id", "style", "cot", "seed", "cfg_scale"):
        value = getattr(args, key)
        if value is not None:
            data[key] = value
    if args.lyrics_file:
        data["lyrics"] = Path(args.lyrics_file).read_bytes().decode("utf-8")
    elif args.lyrics is not None:
        data["lyrics"] = args.lyrics
    if args.abc_file:
        data["abc"] = Path(args.abc_file).read_bytes().decode("utf-8")
        data.pop("abc_path", None)
    if not data:
        data = {"id": "first_song", "style": "Mandarin, warm piano, acoustic pop, female vocal",
                "lyrics": "[Verse]\n晚风轻轻吹过窗前\n你留下的笑还在昨天\n[Chorus]\n让这首歌陪你走远\n把所有想念唱成明天"}
    kwargs = request_kwargs(data, base)
    preflight(kwargs, generation_config(args))
    pipe = get_pipe(args)
    try:
        directory = Path(args.output or "runs/default") / kwargs.get("id", "song")
        if args.resume and (directory / "result.json").exists():
            req_kwargs = {k: v for k, v in kwargs.items() if k not in {"abc_sampling", "semantic_sampling"}}
            request = pipe._request(**req_kwargs)
            config = pipe.effective_config(request, kwargs.get("abc_sampling"), kwargs.get("semantic_sampling"))
            expected = identity({"request": request.to_dict(), "config": config, "weights": pipe.weights})
            result = verify_result(directory, expected)
            print(json.dumps({"resumed": True, "result": str(directory / "result.json"), "truncated": result["truncated"]}))
            return 0
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Nonempty output {directory}; use --resume or a new output directory")
        directory.mkdir(parents=True, exist_ok=True)
        try:
            if args.stage == "plan":
                kwargs.pop("semantic_sampling", None)
                plan = pipe.plan(**kwargs)
                plan.save(directory)
                print(json.dumps({"stage": "plan", "output": str(directory), "truncated": plan.truncated}))
            else:
                result = pipe(**kwargs)
                result.save_artifacts(directory)
                print(json.dumps({"status": "complete", "output": str(directory), "truncated": result.truncated,
                                  "seconds": result.timing["e2e_seconds"]}))
            return 0
        except BaseException as exc:
            write_json(directory / "failure.json", {"status": "failed", "type": type(exc).__name__, "reason": str(exc),
                                                    "request": data})
            raise
    finally:
        pipe.close()


def batch(args):
    if args.concurrency != 1:
        raise ValueError("The minimal torch pipeline currently supports concurrency=1; do not share a pipeline concurrently")
    path = Path(args.input)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    batch_ids(rows)
    defaults = generation_config(args)
    pipe = None
    output = Path(args.output or "runs/batch")
    receipts, failures = [], 0
    try:
        for index, row in enumerate(rows, 1):
            if not args.quiet:
                print(f"Song {index}/{len(rows)}: {row['id']}", file=sys.stderr, flush=True)
            directory = output / row["id"]
            try:
                kwargs = request_kwargs(row, path.parent)
                if args.cot is not None:
                    kwargs["cot"] = args.cot
                request = preflight(kwargs, defaults)
                if pipe is None:
                    pipe = get_pipe(args)
                cfg = pipe.effective_config(request, kwargs.get("abc_sampling"), kwargs.get("semantic_sampling"))
                expected = identity({"request": request.to_dict(), "config": cfg, "weights": pipe.weights})
                if args.resume and (directory / "result.json").exists():
                    receipt = verify_result(directory, expected)
                else:
                    if directory.exists() and any(directory.iterdir()):
                        raise FileExistsError("Nonempty request output; use --resume or a new run")
                    receipt = pipe(**kwargs).save_artifacts(directory)
                receipts.append({"id": row["id"], "status": "complete", "identity": receipt["identity"]})
            except Exception as exc:
                failures += 1
                failure = {"id": row["id"], "status": "failed", "reason": str(exc), "type": type(exc).__name__}
                write_json(directory / "failure.json", failure)
                receipts.append(failure)
            write_json(output / "batch.json", {"complete": len(receipts) == len(rows) and not failures,
                       "expected": len(rows), "failed": failures, "results": receipts})
    finally:
        if pipe is not None:
            pipe.close()
    return int(failures > 0)


def parser():
    p = argparse.ArgumentParser(description="YuE2: style + lyrics → symbolic plan → song")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("doctor", "generate", "batch", "export"):
        q = sub.add_parser(name)
        q.add_argument("--model")
        q.add_argument("--vae", default="standard", help="standard (listening), legacy (paper evaluation), local path or HF repo")
        q.add_argument("--revision")
        q.add_argument("--vae-revision")
        q.add_argument("--device", default="auto")
        q.add_argument("--budget", type=float, default=24)
        q.add_argument("--backend", choices=("torch", "torch-eager", "vllm"), default="torch")
        q.add_argument("--quantization", choices=("none", "fp8"), default="none")
        q.add_argument("--offload-ar", action="store_true")
        q.add_argument("--offline", action="store_true")
        q.add_argument("--config")
        q.add_argument("--output", required=name == "export")
        if name == "export":
            q.add_argument("--quiet", "--no-progress", action="store_true")
        elif name == "doctor":
            q.add_argument("--verify-hashes", action="store_true")
        else:
            q.add_argument("--cot", choices=("full", "melody", "off"))
            q.add_argument("--resume", action="store_true")
            q.add_argument("--quiet", "--no-progress", action="store_true",
                           help="Hide YuE2 progress on stderr; keep result output on stdout")
            if name == "batch":
                q.add_argument("--input", required=True)
                q.add_argument("--concurrency", type=int, default=1)
            else:
                q.add_argument("--request")
                q.add_argument("--id")
                q.add_argument("--style")
                q.add_argument("--lyrics")
                q.add_argument("--lyrics-file")
                q.add_argument("--abc-file")
                q.add_argument("--seed", type=int)
                q.add_argument("--cfg-scale", type=float)
                q.add_argument("--stage", choices=("plan", "audio"), default="audio")
    q = sub.add_parser("inspect", help="Verify saved artifacts without loading models")
    q.add_argument("directory")
    q = sub.add_parser("decode", help="Decode standalone latents with only the VAE")
    q.add_argument("latent")
    q.add_argument("--vae", required=True, help="Local VAE directory or Hugging Face repository")
    q.add_argument("--vae-revision")
    q.add_argument("--offline", action="store_true")
    q.add_argument("--device", default="cpu")
    q.add_argument("--vae-core-frames", type=int, default=1024)
    q.add_argument("--output", required=True)
    q = sub.add_parser("validate", help="Check request JSON or batch JSONL without models")
    source = q.add_mutually_exclusive_group(required=True)
    source.add_argument("--request")
    source.add_argument("--input")
    q.add_argument("--config")
    q.add_argument("--output")
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parser().parse_args(argv)
    commands = {"doctor": doctor, "generate": generate, "batch": batch, "export": export_models,
                "inspect": inspect_run, "decode": decode_latents, "validate": validate_requests}
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
