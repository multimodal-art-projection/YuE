#!/usr/bin/env python3
"""Generate a verified instrumental request and export a duration-limited preview."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--low-vram", action="store_true")
    args = parser.parse_args()
    if not math.isfinite(args.seconds * 48000) or args.seconds <= 0 or round(args.seconds * 48000) < 1:
        parser.error("seconds must be finite, positive and at least one audio sample")
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    if not (args.prepared / "prepared-manifest.json").is_file():
        parser.error("Use the instrumental workflow's prepare command first")
    helper = Path(__file__).resolve().parents[1] / "skills/yue2-music/instrumental/scripts/instrumental.py"
    subprocess.run([sys.executable, str(helper), "verify", str(args.prepared)], check=True)
    request = json.loads((args.prepared / "request.json").read_text(encoding="utf-8"))
    from yue2 import YuE2Pipeline
    from yue2.storage import write_json
    import soundfile as sf

    with YuE2Pipeline.from_pretrained(device=args.device, low_vram=args.low_vram) as pipe:
        song = pipe(**request)
        song.save_artifacts(args.output / "native")
    samples = round(args.seconds * song.sample_rate)
    clip = song.audio[:samples]
    sf.write(args.output / "clip.wav", clip, song.sample_rate, subtype="FLOAT")
    record = {"requested_seconds": args.seconds, "clip_seconds": len(clip) / song.sample_rate,
              "source_seconds": len(song.audio) / song.sample_rate,
              "cropped": len(clip) < len(song.audio), "shorter_than_requested": len(clip) < samples,
              "generation_truncated": song.truncated}
    write_json(args.output / "clip.json", record)
    print(json.dumps(record))
    return int(any(song.truncated.values()) or len(clip) < samples)


if __name__ == "__main__":
    raise SystemExit(main())
