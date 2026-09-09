# Edit with a score and an agent

YuE2 makes the intended composition available as ABC notation. An agent can change specific notes, chord symbols, tempo, or section order, then use the same generator to render the revised composition. This is the editable interface meant by **white-box music generation**.

## A complete harmony-edit example

The original fixtures keep the exact melody, lyric order, meter, tempo, style, and seed fixed. `score-jazz.abc` changes the chord symbols in `score.abc` to seventh chords.

```bash
python skills/yue2-music/scripts/abc_tools.py inspect examples/score.abc
python skills/yue2-music/scripts/abc_tools.py inspect examples/score-jazz.abc
python skills/yue2-music/scripts/abc_tools.py compare \
  examples/score.abc examples/score-jazz.abc
python examples/generate.py --abc-file examples/score.abc \
  --cot full --output outputs/harmony-before
python examples/generate.py --abc-file examples/score-jazz.abc \
  --cot full --output outputs/harmony-after
```

The symbolic comparison should report `match: true`: notes, timing, meter, and tempo are unchanged. That check deliberately allows different harmony. Listen to both outputs to judge whether the revised harmony is realized and whether you prefer it.

## Start from a generated composition

Generate a song with `save_artifacts()` to retain both its audio and score, or export only the plan:

```bash
python skills/yue2-music/scripts/run_yue2.py plan \
  --request examples/song.json --output outputs/plan
cp outputs/plan/score.abc edited.abc
```

Give the agent the ABC, lyrics, style, and a bounded brief. For example:

> Reharmonize this song with jazz-influenced chords. Preserve every vocal and instrumental melody note, note duration, bar boundary, section order, and tempo. Keep the lyrics and style unchanged for this comparison. Save a new ABC file and explain the chord changes.

Then inspect the edited score and verify the requested invariants:

```bash
python skills/yue2-music/scripts/abc_tools.py inspect edited.abc
python skills/yue2-music/scripts/abc_tools.py compare outputs/plan/score.abc edited.abc
python examples/generate.py --request examples/song.json \
  --abc-file edited.abc --cot full --output outputs/edited
```

The helper checks a limited native ABC dialect. A rejected score may need notation conversion or correction; do not treat an unsupported construct as a demonstrated musical error. See the [ABC reference](../skills/yue2-music/references/abc-editing.md).

## Edit lyrics, style, tempo, or form

Copy the request JSON and update only the intended fields. Align revised lyrics with the melody's phrasing and syllable counts. For section deletion, duplication, or reordering, update both the score and corresponding lyric sections. To change tempo while preserving notes, use the comparison helper's `--allow-tempo-change` option; listen to verify the actual response.

Editing renders a new complete recording. Matching the unchanged score does not guarantee identical singing, timbre, or waveform outside the edit. Preserve the original and use fresh output directories for every version. Check truncation and compare complete outputs rather than choosing examples solely by the desired result.

## Use the packaged skill

Load [yue2-music](../skills/yue2-music/SKILL.md) in an agent supporting `SKILL.md` packages. Its helpers cover generation, transcription, ABC checks, cached decoding, and listening comparisons. The skill guides the agent between existing APIs; no separate agentic-generation checkpoint is required.

[Explore the 9-step, 14-version editing demo](https://map-yue2.github.io/#agentic-music-editing), where each version includes its audio, score, prompts, lyrics, and conversation.
