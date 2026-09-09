#!/usr/bin/env python3
"""Generate, plan, or re-decode with yue2-infer 0.1.6; preserve native artifacts."""

import argparse
import importlib.metadata
import sys
import time
from pathlib import Path

from abc_tools import parse_abc, report
from common import fresh_directory, read_json, sha256, write_json


def request_data(args):
    data = read_json(args.request)
    if not isinstance(data, dict):
        raise ValueError("Request must be a JSON object")
    allowed = {"style", "tags", "lyrics", "cot", "seed", "abc", "abc_path", "cfg_scale", "id"}
    if set(data) - allowed:
        raise ValueError(f"Unsupported request fields: {sorted(set(data) - allowed)}")
    if "tags" in data:
        tags = data.pop("tags")
        if "style" in data and data["style"] != tags:
            raise ValueError("style and tags disagree")
        data["style"] = tags
    sources = sum(x is not None for x in (data.get("abc"), data.get("abc_path"), args.abc_file))
    if sources > 1:
        raise ValueError("Use only one of abc, abc_path or --abc-file")
    abc_path = data.pop("abc_path", None)
    if abc_path is not None:
        data["abc"] = (Path(args.request).parent / abc_path).read_bytes().decode("utf-8")
    if args.abc_file:
        data["abc"] = Path(args.abc_file).read_bytes().decode("utf-8")
    if args.cot:
        data["cot"] = args.cot
    data.setdefault("cot", "full")
    data.setdefault("id", "song")
    if args.action == "all-modes" and data.get("abc") is not None:
        raise ValueError("all-modes requires text-only input, since off cannot accept ABC")
    if args.action == "plan" and data["cot"] == "off":
        raise ValueError("off has no symbolic plan; use generate")
    if data.get("abc") is not None:
        if data["cot"] == "off":
            raise ValueError("off cannot accept ABC")
        score = parse_abc(data["abc"])
        if data["cot"] == "melody" and any(v.chords for v in score.voices.values()):
            raise ValueError("melody input still contains chords; use abc_tools.py strip-chords")
    return data


def options(args):
    return dict(model=args.model, vae=args.vae, revision=args.revision,
                vae_revision=args.vae_revision, local_files_only=args.offline,
                device=args.device, memory_budget_gib=args.memory_budget_gib)


def score_check(text, mode):
    if text is None:
        return {"status": "not_applicable"}
    try:
        score = parse_abc(text)
        if mode == "melody" and any(v.chords for v in score.voices.values()):
            raise ValueError("Melody-mode planner returned chord symbols")
        return {"status": "passed", "scope": "native ABC structure only", "score": report(score)}
    except ValueError as exc:
        return {"status": "failed", "error": str(exc)}


def decode(args, output, loader):
    import numpy as np
    from yue2 import SemanticResult, SongResult, SymbolicPlan, YuE2Pipeline
    from yue2.storage import identity, verify_result

    source = Path(args.source)
    verify_result(source)
    original = read_json(source / "result.json")
    original_config = read_json(source / "config.json")
    plan = SymbolicPlan.load(source)
    tokens = np.load(source / "semantic.npy", allow_pickle=False)
    if tokens.ndim != 1 or tokens.dtype.kind not in "iu":
        raise ValueError("Invalid cached semantic tokens")
    latents = np.load(source / "latent.npy", allow_pickle=False)
    semantic = SemanticResult(plan, tokens.tolist(), original["timing"].get("semantic", {}),
                              original["truncated"]["semantic"])
    with YuE2Pipeline.from_pretrained(**loader) as pipe:
        if pipe.weights["mot"] != original["weights"]["mot"]:
            raise ValueError("--model must match the source generation's exact model identity")
        start = time.perf_counter()
        audio = pipe.decode(latents)
        elapsed = time.perf_counter() - start
        current = pipe.effective_config(plan.request)
        config = dict(original_config)
        for key in ("vae_dtype", "vae_decode", "vae_core_frames", "vae_halo_frames", "decoder_release"):
            config[key] = current[key]
        config["cached_decode"] = {
            "source_identity": original["identity"],
            "source_config_sha256": sha256(source / "config.json"),
            "runtime_sha256": pipe.runtime_sha256,
            "device": str(pipe.device), "memory_budget_gib": pipe.memory_budget_gib,
        }
        # Top-level generation/runtime fields describe the cached source generation.
        # cached_decode describes this new decoder execution separately.
        timing = {"operation": "decode_cached_latents", "vae_seconds": elapsed,
                  "semantic": semantic.timing, "source_generation": original["timing"]}
        write_json(output / "source_generation.json", {
            "source_result_sha256": sha256(source / "result.json"),
            "source_latent_sha256": sha256(source / "latent.npy"),
            "source_semantic_sha256": sha256(source / "semantic.npy"),
            "config": original_config, "identity": original["identity"],
            "weights": original["weights"],
        })
        digest = identity({"request": plan.request.to_dict(), "config": config, "weights": pipe.weights})
        song = SongResult(audio, 48000, semantic, latents, config, pipe.weights, timing, digest)
        receipt = song.save_artifacts(output)
    verify_result(output)
    if sha256(output / "latent.npy") != sha256(source / "latent.npy"):
        raise ValueError("Latents changed during re-decoding")
    return {"status": "complete", "identity": receipt["identity"],
            "truncated": receipt["truncated"], "audio_seconds": receipt["audio_seconds"]}


def run(args):
    data = None if args.action == "decode" else request_data(args)
    # Import only after cheap request/ABC preflight, and before reserving output.
    from yue2 import YuE2Pipeline
    from yue2.protocol import SongRequest
    from yue2.storage import verify_result

    requests = []
    if data is not None:
        if args.action == "all-modes":
            requests = [SongRequest(**dict(data, cot=m, id=f"{data['id']}_{m}"))
                        for m in ("full", "melody", "off")]
        else:
            requests = [SongRequest(**data)]
    output = fresh_directory(args.output)
    loader = options(args)
    write_json(output / "invocation.json", {
        "action": args.action, "loader": loader,
        "package_version": importlib.metadata.version("yue2-infer"),
        "requests": [r.to_dict() for r in requests],
    })
    try:
        if args.action == "decode":
            result = decode(args, output, loader)
            write_json(output / "run.json", result)
            return int(any(result["truncated"].values()))
        results = []
        with YuE2Pipeline.from_pretrained(**loader) as pipe:
            for request in requests:
                destination = fresh_directory(output / request.cot) if args.action == "all-modes" else output
                write_json(destination / "input.json", request.to_dict())
                try:
                    if args.action == "plan":
                        plan = pipe.plan(request=request)
                        plan.save(destination)
                        write_json(destination / "request.json", request.to_dict())
                        write_json(destination / "provenance.json", {
                            "weights": pipe.weights, "config": pipe.effective_config(request),
                            "loader": loader,
                        })
                        check = score_check(plan.abc, request.cot)
                        result = {"mode": request.cot, "truncated": {"abc": plan.truncated}}
                    else:
                        song = pipe(**request.to_dict())
                        receipt = song.save_artifacts(destination)
                        verify_result(destination)
                        check = score_check(song.abc, request.cot)
                        result = {"mode": request.cot, "identity": receipt["identity"],
                                  "truncated": receipt["truncated"], "audio_seconds": receipt["audio_seconds"]}
                    write_json(destination / "abc_check.json", check)
                    result["status"] = "needs_review" if any(result["truncated"].values()) or check["status"] == "failed" else "complete"
                except Exception as exc:
                    result = {"mode": request.cot, "status": "failed", "error": str(exc), "type": type(exc).__name__}
                    write_json(destination / "failure.json", result)
                results.append(result)
                print(result, flush=True)
        write_json(output / "run.json", {"results": results})
        return int(any(r["status"] != "complete" for r in results))
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "error": str(exc), "type": type(exc).__name__})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for action in ("generate", "all-modes", "plan", "decode"):
        command = sub.add_parser(action)
        command.add_argument("--output", required=True, type=Path, help="Fresh destination; no automatic overwrite/resume")
        if action == "decode":
            command.add_argument("--source", required=True, type=Path, help="Verified native SongResult directory")
        else:
            command.add_argument("--request", required=True, type=Path)
            command.add_argument("--cot", choices=("full", "melody", "off"))
            command.add_argument("--abc-file", type=Path)
        command.add_argument("--model", default="m-a-p/YuE2-3B")
        command.add_argument("--vae", default="m-a-p/YuE2-Vae")
        command.add_argument("--revision")
        command.add_argument("--vae-revision")
        command.add_argument("--offline", action="store_true")
        command.add_argument("--device", default="cuda")
        command.add_argument("--memory-budget-gib", type=float, default=24)
    args = parser.parse_args()
    try:
        return run(args)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
