# Original examples

`song.json` contains the original **City Lights** lyrics and style request used by the agent skill. `melody.abc` is a short, original eight-bar melody for those words. Each lyric line has seven syllables and each bar has seven note onsets. `score.abc` adds harmony; `score-jazz.abc` changes only the chord symbols. These are runnable inputs, not recorded benchmark or quality results.

From the repository root, after `pip install .`:

```bash
# Create a song and its score from lyrics and style.
python examples/generate.py --output outputs/create

# Realize the included melody with free accompaniment.
python examples/generate.py --abc-file examples/melody.abc \
  --cot melody --output outputs/melody

# Compare two harmonizations with the same melody, lyrics, style, and seed.
python examples/generate.py --abc-file examples/score.abc \
  --cot full --output outputs/harmony-original
python examples/generate.py --abc-file examples/score-jazz.abc \
  --cot full --output outputs/harmony-edited
```

The script preserves all native artifacts, refuses an existing output directory, and exits nonzero if generation reports truncation. The seed is fixed in `song.json`; seeds aid comparison but do not guarantee identical output across devices or software versions. Use `--revision` and `--vae-revision` to pin model versions.

Check the symbolic intervention without a GPU:

```bash
python skills/yue2-music/scripts/abc_tools.py inspect examples/score.abc
python skills/yue2-music/scripts/abc_tools.py compare \
  examples/score.abc examples/score-jazz.abc
```

The comparison verifies unchanged note pitches, timing, meter, and tempo. Listening or audio transcription is still needed to assess the realized music.
