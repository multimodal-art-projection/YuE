# Musical editing and agent delegation

Treat an edit as a new musical version with a preserved source. YuE2 regenerates audio from the revised score, style and lyrics; it does not expose waveform inpainting or guarantee an identical performance outside the edited bars. Use [generation-and-covers.md](generation-and-covers.md) for the calls and [abc-editing.md](abc-editing.md) for notation and event checks.

## Start with an audible baseline and an explicit contract

Generate a full plan and render its original song before editing. Retain the exact request, plan tokens, ABC, semantic tokens, acoustic latents and decoder identity. For a recording, retain its raw SheetSage2 transcription as well as the corrected score used for generation.

Define what “preserve the melody” means before delegating:

| Contract | What must remain fixed |
|---|---|
| Exact melody and rhythm | Sounding pitch, note onset and duration in the specified voices/passages |
| Same pitches, revised rhythm | Ordered sounding pitches; record the allowed rhythm and articulation changes |
| Recognizable theme | Complete phrase sequence and specified repetitions; list allowed embellishments |
| Free adaptation within bounds | Agreed phrases, range, contour, cadences or motifs; report actual changed notes |

Specify whether tempo, meter, lyrics and instrumental passages may change. A tempo change preserves beat-relative rhythm but changes time in seconds. A chord-only edit may add tied ABC tokens while preserving every sounding note. Compare interpreted events, not text similarity.

For a style change, update the arrangement prompt along with the score. For example, a relaxed modern jazz version can request piano, tenor saxophone, upright bass, brushed drums, no guitar and 88 BPM. Put `Q:1/4=88` in the score too. These are example musical choices, not mandatory defaults for every edit. Describing an instrument omission does not prove that the generated audio omits it.

## Delegate a bounded musical task

Use [assets/edit-brief.md](../assets/edit-brief.md). Give the editing agent actual input files and user requirements, not a summary that omits inconvenient passages. It should receive:

- Untouched ABC, exact style/lyrics and the baseline audio when audio review is available.
- Requested changes and a contract naming the voices, sections and permitted differences.
- Native notation limits and the required output directory.

Require a complete edited score, revised request, edit manifest and a brief musical rationale. The manifest should identify source hashes, changed bar/beat ranges, chord boundaries, note/rhythm changes, lyric changes and preserved invariants. The agent must distinguish score work from audio it actually generated or heard.

Give an independent reviewer the raw before/after ABC, requests, the user's constraints and available audio. Ask it to identify violations and musical problems from these artifacts before revealing the editor's diagnosis or rationale. Do not lead it with “the problem is chord X” or ask it merely to endorse the proposed repair. After its independent pass, reconcile the findings with the editor's change record.

If no delegation mechanism is available, perform the editing and review passes explicitly yourself. Do not invent an agent call. CPU score review can run alongside other useful work; model generation stays sequential within each YuE2 pipeline.

## Reharmonize by listening to note spans

First mark sustained notes, strong-beat notes, cadences and phrase boundaries in **both** voices. For every candidate change, inspect the active chord throughout the note, the bass movement, adjacent melody pitches and how the tension resolves. A two-beat collision needs a different judgment from a short chromatic approach.

Introduce modern color with a musical direction: secondary dominants into temporary tonal centers, a short ii–V, borrowed minor harmony, a tritone approach, or a common-tone connection. Keep a coherent bass line and guide-tone motion. More substitutions per bar do not automatically create a better jazz arrangement.

Preserve colors that work and repair the smallest relevant span. Options include moving a resolution earlier, choosing a chord core that supports a held melody pitch, changing a bass inversion, or adapting one melodic approach when the contract permits it. An altered dominant's available tensions are not a blanket excuse for any sustained melody–voicing collision. Conversely, do not replace every non-chord tone with a chord tone.

Use only native chord symbols. Extended voicing requests belong in the style prompt when their symbols are outside the released notation vocabulary. Keep the original voices, section comments, rhythmic unit and bar grouping. Do not add a piano chord staff, lyric `w:` lines, custom phoneme annotations or unsupported chord suffixes. A format check and a harmony judgment are separate requirements.

## Worked arrangement brief: complete theme, modern repeat, solo

A useful example is a pop-to-jazz adaptation with a complete “Twinkle, Twinkle, Little Star” theme stated twice, followed by a tenor saxophone solo. The full six-phrase scale-degree melody is:

```text
1 1 5 5 6 6 5
4 4 3 3 2 2 1
5 5 4 4 3 3 2
5 5 4 4 3 3 2
1 1 5 5 6 6 5
4 4 3 3 2 2 1
```

That is **42 sounding notes per statement, 84 for two complete statements**. These are phrases, not ABC bar lines. In D major, the opening phrase is D–D–A–A–B–B–A, and the next is G–G–F♯–F♯–E–E–D. Transpose using the active key; do not copy those absolute pitches into an unrelated key.

For this brief:

1. Lead into the quotation with a short bridge and a melodic pickup sharing notes with the theme. Adjust the sung words to leave a natural breath and a clear handoff to saxophone.
2. State all six phrases once with clear rhythm and supportive harmony. A conventional quarter-quarter-quarter-quarter-quarter-quarter-half rhythm makes each phrase two 4/4 bars; other rhythms must be intentional.
3. State all six phrases a second time with more adventurous harmony. Keep the melody recognizable, use carefully prepared tonicizations or substitutions, and resolve sustained tensions. If changing the rhythm, retain the full pitch sequence and audit the permitted differences.
4. Develop the material into an eight-bar jazz solo: vary rhythm and register, answer fragments, then move toward a cadence returning to the song. Repeating the opening seven notes is not a substitute for either complete statement.
5. Keep the relaxed 88 BPM arrangement, piano, tenor saxophone, upright bass and brushes. If a short 3/4 transition helps, make the change explicit in both native voice blocks and verify the return to 4/4. Do not insert odd meters merely to make the arrangement seem advanced.

Place the instrumental theme/solo in the native `Ins` melody voice and put appropriate rests in `Vocal`. Retain harmony symbols in the native harmony-bearing voice even during vocal rests. A lyric tag such as `[saxophone solo]` and a matching style description help communicate intent, but are not a sample-accurate scheduling API. Validate all 84 merged note events before assessing the added solo separately.

## Adapt lyrics for singing

Translate meaning into singable lines rather than forcing a literal translation onto the old notes. Start from the final vocal event grid, including pickups, ties, rests and melismas. Check syllable count, lexical stress, consonant density on short notes, comfortable vowels on held notes and breathing spaces. Revise wording or rhythm where allowed; matching the total number of syllables alone is insufficient.

Keep a pronunciation sidecar with an explicit note-index convention, selected pronunciation source and any manual choices. For example:

```json
{
  "voice": "Vocal",
  "note_index_base": 0,
  "word": "light",
  "syllable": "light",
  "phonemes": ["L", "AY1", "T"],
  "stress": 1,
  "note_indices": [20, 21],
  "articulation": "Attack L once; sustain AY across both notes; release T at the end"
}
```

This illustrates intended alignment, not a measured transcript or a YuE2 input field. Account for every intended vocal note and syllable; one syllable can span several notes, while its consonants and vowel have different roles. Record the actual dictionary pronunciation rather than guessing phonemes, and flag out-of-vocabulary words for review. Do not alter the native ABC format to insert this information.

## Render, compare and iterate

Run structural checks and the requested event comparisons before inference. Keep the original score and every attempted revision. If the contract permits rhythm or melody changes, review the explicit differences rather than labeling a failed exact comparison “preserved.”

Render the changed version with a new request and output directory. Listen to full songs as well as excerpts beginning before and ending after the changed passage. Check the entry into the theme, its complete second statement, solo development, return to vocals and English diction where relevant. A same-seed comparison is useful provenance, not a guarantee of controlled acoustic variation.

If a requested property fails in the audio, revise and rerender it. Report what is verified symbolically, what was observed by listening, and what remains uncertain. Follow [listening-and-evaluation.md](listening-and-evaluation.md) to deliver the audio, actual prompts, score changes and evaluation evidence together.
