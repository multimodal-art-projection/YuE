#!/usr/bin/env python3
"""Transcribe audio to native ABC using SheetSage2's Transformers interface."""

import argparse
import importlib.metadata
import inspect
import sys
from pathlib import Path

from abc_tools import parse_abc, report
from common import fresh_directory, sha256, write_json


def run(args):
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    if args.max_seconds is not None and args.max_seconds <= 0:
        raise ValueError("--max-seconds must be positive and explicitly crops the input")
    output = fresh_directory(args.output)
    melody_only = args.task != "full"
    prompts = ["timestamp", "downbeat_meter", "structure", "key"]
    if args.task == "full":
        prompts += ["chord_full", "melody_full"]
    else:
        prompts += ["melody_vocal" if args.task == "melody-vocal" else "melody_full"]
    write_json(output / "input.json", {
        "source_name": args.audio.name, "source_audio_sha256": sha256(args.audio),
        "model": args.model, "revision": args.revision, "offline": args.offline,
        "base_model_path": args.base_model, "prompts": prompts, "melody_only": melody_only,
        "preset": args.preset, "max_seconds": args.max_seconds,
        "device": args.device, "dtype": args.dtype,
    })
    try:
        import torch
        from transformers import AutoModel

        torch.set_num_threads(args.threads)
        loader = dict(trust_remote_code=True, local_files_only=args.offline)
        if args.revision:
            loader.update(revision=args.revision, code_revision=args.revision)
        if args.base_model:
            loader["base_model_path"] = args.base_model
        model = AutoModel.from_pretrained(args.model, **loader).eval().to(args.device)
        options = {}
        if melody_only:
            try:
                parameter = inspect.signature(model.transcribe).parameters.get("melody_only")
            except (TypeError, ValueError) as exc:
                raise RuntimeError("Cannot verify SheetSage2's melody_only interface; refresh the model code "
                                   "to a reviewed revision exposing melody_only explicitly") from exc
            if parameter is None or parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
                raise RuntimeError("This SheetSage2 revision does not expose melody_only; refresh the model "
                                   "and remote code to a reviewed revision supporting melody_only=True")
            options["melody_only"] = True
        snapshot = Path(getattr(model, "_source_snapshot", args.model))
        files = {}
        if snapshot.is_dir():
            for path in sorted(snapshot.iterdir()):
                if path.is_file() and (path.suffix in {".py", ".safetensors"} or path.name == "config.json"):
                    files[path.name] = sha256(path)
        write_json(output / "model_provenance.json", {
            "model": args.model, "requested_revision": args.revision,
            "config": model.config.to_dict(), "snapshot_sha256": files,
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "huggingface-hub")},
        })
        result = model.transcribe(
            str(args.audio), output_dir=str(output), prompts=prompts,
            dtype=args.dtype, preset=args.preset, max_seconds=args.max_seconds, **options,
        )
        if result.get("abc_error") or not result.get("abc"):
            raise ValueError(f"Transcription produced no usable ABC: {result.get('abc_error')}")
        score = parse_abc(result["abc"])
        if args.task != "full" and any(v.chords for v in score.voices.values()):
            raise ValueError("Melody transcription contains unexpected chord symbols")
        if not (output / "score.abc").is_file():
            raise ValueError("Transcriber did not save score.abc")
        write_json(output / "abc_check.json", {"status": "passed", "score": report(score),
                   "scope": "symbolic format; transcription accuracy still needs review"})
        write_json(output / "transcription_manifest.json", {
            "status": "complete", "warnings": result.get("warnings", []),
            "source_audio_sha256": sha256(args.audio),
            "artifacts": {str(p.relative_to(output)): sha256(p)
                          for p in sorted(output.rglob("*")) if p.is_file()},
        })
        print(f"Saved {output / 'score.abc'}; warnings: {result.get('warnings', [])}")
    except Exception as exc:
        write_json(output / "failure.json", {"status": "failed", "type": type(exc).__name__, "error": str(exc)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Fresh output directory")
    parser.add_argument("--task", choices=("full", "melody-full", "melody-vocal"), default="full")
    parser.add_argument("--model", default="m-a-p/SheetSage2")
    parser.add_argument("--revision", help="Pin model and remote code to the same commit")
    parser.add_argument("--base-model", help="Verified MERT-v2-FullSong snapshot for an offline adapter load")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--preset", choices=("default", "paper"), default="default")
    parser.add_argument("--max-seconds", type=float, help="Explicitly crop audio; omitted means process the whole input")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        run(args)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
