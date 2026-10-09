#!/usr/bin/env python3
"""Render a short instrumental clip with the instrumental-skill recipe, in one model load.

YuE2 writes a score, the instrumental skill moves the Vocal melody onto the
instrumental part, and the stock model renders the revised score. Compared with
running the skill's `instrumental.py run` per clip, this loads the model once and
caps the semantic token budget to the length you need (about 25 tokens per second)
instead of always rendering a full-length song, which suits batches of short clips.
"""

import argparse
import dataclasses
import json
import re
import sys
from pathlib import Path

SKILL_SCRIPTS = Path(__file__).resolve().parents[1] / "skills/yue2-music/instrumental/scripts"
TOKENS_PER_SECOND = 25
PLANNING_LYRICS = "[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n"


def instrumental_style(style):
    """Same style normalization the skill applies before rendering."""
    style = style.strip().rstrip(".,")
    if not re.match(r"^instrumental\b", style, re.I):
        style = "Instrumental, " + style
    for condition in ("no vocals", "no singing", "no choir", "no spoken words"):
        if condition not in style.lower():
            style += ", " + condition
    return style + "."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--style", required=True, help="Short description of the music")
    parser.add_argument("--seconds", type=float, default=60.0, help="Length to keep and to budget tokens for")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True, help="Output .flac path")
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--vae", default="m-a-p/YuE2-Vae")
    parser.add_argument("--save-score", type=Path, help="Optional path for the transferred instrumental ABC score")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output path to retain each version.")

    sys.path.insert(0, str(SKILL_SCRIPTS))
    import instrumental as skill
    import soundfile as sf
    from instrumentalize import convert_score
    from yue2 import YuE2Pipeline
    from yue2.protocol import SongRequest

    with YuE2Pipeline.from_pretrained(args.model, vae=args.vae, device="cuda") as pipe:
        planned = pipe.plan(request=SongRequest(id="clip", style=args.style, lyrics=PLANNING_LYRICS,
                                                cot="full", seed=args.seed))
        if planned.truncated or not planned.abc:
            raise SystemExit("The model's score was empty or truncated; try another seed.")
        text, _ = convert_score(planned.abc, overlap="vocal", keep_chords=True)
        score = skill.validate_score(text)
        request = dict(id="clip", style=instrumental_style(args.style), lyrics=skill.lyric_tags(text), abc=text,
                       cot="full" if score.voices["Vocal"].chords else "melody", seed=args.seed)
        skill.validate_request(request)
        if args.save_score:
            args.save_score.write_text(text, encoding="utf-8")
        plan = pipe.plan(**request)
        budget = dataclasses.replace(pipe.generation_config.semantic,
                                     max_tokens=int(args.seconds * TOKENS_PER_SECOND) + 250)
        semantic = pipe.generate_semantic(plan, sampling=budget)
        audio = pipe.decode(pipe.synthesize(semantic))

    audio = audio[: int(args.seconds * 48000)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, audio, 48000)
    print(json.dumps({"audio": str(args.output), "seconds": round(len(audio) / 48000, 1)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
