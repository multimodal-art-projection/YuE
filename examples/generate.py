#!/usr/bin/env python3
"""Generate from original example lyrics, optionally with a supplied ABC score."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, default=Path(__file__).with_name("song.json"))
    parser.add_argument("--abc-file", type=Path)
    parser.add_argument("--cot", choices=("full", "melody", "off"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae")
    parser.add_argument("--revision")
    parser.add_argument("--vae-revision")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory to retain each version.")
    request = json.loads(args.request.read_text(encoding="utf-8"))
    if args.abc_file:
        request["abc"] = args.abc_file.read_text(encoding="utf-8")
    if args.cot:
        request["cot"] = args.cot
    if request.get("abc") is not None and request.get("cot", "full") == "off":
        parser.error("A supplied score requires full or melody mode.")
    from yue2 import YuE2Pipeline

    with YuE2Pipeline.from_pretrained(
        args.model, vae=args.vae, revision=args.revision,
        vae_revision=args.vae_revision, device="cuda",
    ) as pipe:
        song = pipe(**request)
        song.save_artifacts(args.output)
        print(json.dumps({"audio": str(args.output / "audio.flac"), "truncated": song.truncated}))
        return 1 if any(song.truncated.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
