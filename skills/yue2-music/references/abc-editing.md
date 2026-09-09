# Edit the score while preserving its musical meaning

Start from `plan.save(...)`'s `score.abc`, or SheetSage2's exported `score.abc` when covering an audio recording. Keep an untouched source and write a new edited file. Model-generated plans and transcriptions can contain mistakes; inspect them before treating them as a musical reference.

The helper in this skill is an original Python standard-library implementation for a **bounded native ABC dialect**. It checks structure and exact symbolic melody. It does not implement the entire ABC standard, force the generator to follow the score, or measure perceptual harmony.

## Native format

Preserve the source's format, key, register, voices and section boundaries. The key is not fixed to D. A typical header is:

```abc
X:1
T:
M:4/4
L:1/32
Q:1/4=88
V: Vocal clef=treble name="Vocal Melody" snm="Vocal"
V: Ins clef=treble name="Ins Melody" snm="Inst."
K:G
% verse
V: Vocal
"Gmaj7"B8d8"Am7"c8A8|"D7"F16"G"G16|
V: Ins
Z2|
```

This is a two-bar format example, not a complete song. Both parts are monophonic melody lines. The `Ins` part can carry an instrumental theme or solo; it is not a piano chord-voicing staff. Harmony is represented by quoted symbols in `Vocal`, including when that part is resting.

The native exporter groups one to four measures at a time, with a `V: Vocal` block followed by a `V: Ins` block. `% verse`, `% chorus`, `% bridge`, `% interlude` and similar comments delimit structure. A meter or key change starts a new group, with matching `M:` or `K:` fields in both voice blocks. An inline `[K:...]` change can occur inside a measure, but both voices must change key at the same musical time.

`L:1/32` is common, not universal: the native exporter chooses the denominator needed by its rhythmic grid. Preserve an exported `L:` value. The helper accepts `L:1/<power of two>` through 1/1024, explicit fractional meters such as `3/4`, `6/8` and `7/8`, standard major/minor keys, and a positive integer quarter-note tempo.

With `L:1/32`:

| Notation | Meaning |
|---|---|
| `C` / `C2` / `C8` | C4 for one thirty-second / one sixteenth / one quarter note |
| `c`, `c'`, `C,` | C5, C6, C3 |
| `z8` | Quarter-note rest |
| `Z`, `Z2`, `Z3`, `Z4` | One, two, three, four complete resting measures in the current meter |
| `C8-C8` | One half-note C, with no new attack on the second token |
| `"Am7"z16` | A minor seventh chord starts here; the melody rests for half a note |
| `^F`, `_B`, `=F`, `^^F`, `__B` | Sharp, flat, natural, double sharp, double flat |

A 4/4 bar totals 32 units; 3/4 and 6/8 each total 24; 3/8 totals 12. Full-bar `Z` rests count by measures, not by `L:` units. Do not use a compressed rest over a harmonic or key change: retain ordinary rests with the event at its correct onset.

The supported duration multipliers are `1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48`. Express 10 units as `C8-C2`, or `z8z2` for a rest. A chord change inside a held note should split the notation with a tie, for example `"C"E16-"Am7"E16`; otherwise the second `E` is a new attack. A rest cannot be tied, a tie must join equal sounding pitches, and a score cannot end with an unresolved tie.

## Accidentals and ties need musical interpretation

Pitches are relative to the active key. In `K:D`, an unmarked `F` sounds F-sharp and `C` sounds C-sharp. Accidentals remain active through the bar and reset at its boundary or at a key change. **The native export/render convention propagates an accidental by letter across octaves**: after `^F`, both subsequent `F` and `f` are sharp in that bar. This differs from some general-purpose ABC implementations, so do not silently substitute a parser with different accidental semantics.

A tied, unmarked continuation across a barline retains the preceding note's pitch. Its following, untied notes use the new bar's accidental state. For example, in C major, `^F32-|F8F24|` represents a five-quarter F-sharp followed by a three-quarter F-natural. Repeating the explicit accidental on the continuation is clearer and matches the native writer when needed. An explicit contradictory accidental must fail validation.

Count **sounding notes after merging ties**, not raw ABC note tokens. A chord-only edit can add tied tokens without adding melody notes. Conversely, replacing `C8-C8` with `C8C8` changes articulation even though the pitch sequence looks similar.

## Portable inspection and cover preparation

Run these commands from this skill's directory, or adjust the script path:

```bash
python scripts/abc_tools.py inspect source.abc --output source-inspection.json
python scripts/abc_tools.py strip-chords source.abc cover.abc
python scripts/abc_tools.py compare source.abc cover.abc
```

`strip-chords` validates the input and output, removes only supported quoted chord symbols from music lines, and verifies that all sounding notes, onsets, durations, meters and tempo remain unchanged. Quoted voice names in the header are preserved. It refuses to overwrite an existing output.

The default keeps **both** melodies, including instrumental themes and solos. If the cover should preserve only one part, make that selection explicit:

```bash
python scripts/abc_tools.py strip-chords source.abc vocal-cover.abc --keep-voice Vocal
python scripts/abc_tools.py compare source.abc vocal-cover.abc --voices Vocal
```

The unselected voice is replaced with rests on the same time grid; the two-voice format remains intact. Choosing `Ins` instead retains the instrumental melody. Do not discard `Ins` merely because a task is called a cover. Supply the resulting chord-free ABC with `cot="melody"`, the target style and the target lyrics; see [generation and covers](generation-and-covers.md).

Inspection JSON includes exact quarter-note fractions, MIDI pitches, merged note durations, per-voice bar grids, chord onsets, key changes and nominal duration. An API caller can import `parse_abc(text)`; the result exposes `.voices["Vocal"].chords` and `.voices["Ins"].chords` without loading any models.

The helper rejects unsupported material rather than assigning it guessed timing: tuplets, grace notes, polyphonic note stacks, repeat signs, alternate endings, slurs, broken rhythms, decorations, lyric `w:` fields, custom voices/directives, and unsupported key modes or chord qualities. Rebuild such material into the native dialect deliberately. A rejection can mean “outside this helper's scope”; it does not establish that a score is invalid under the full ABC standard.

## Reharmonization and selective melodic editing

The native quoted chord vocabulary is:

```text
major (no suffix), m, dim, aug, 7, maj7, m7, dim7, m7b5,
sus4, sus2, 6, m6, 7sus4, m(maj7)
```

Roots and optional slash basses use note names: `Dbaug`, `F#m7/C#`, `Gm6/Bb`, `A7/E`. Double accidentals are supported where a spelling requires them. `C:maj`, numbered slash degrees, `C13`, `A7alt`, `Cmaj9` and arbitrary annotations are not native chord symbols. Do not invent syntax to request advanced harmony. Use supported chord cores and describe voicing choices in the style prompt; the prompt's richer voicings remain a generative request, not a guarantee.

Before editing, write a short change contract: which voices, sections, melody pitches, rhythms, lyric text and tempo are fixed, and which may change. Give an editing agent the source ABC, style, lyrics and that contract. Require an edited ABC file, updated style, a per-bar explanation of changes, and an invariant-check result. Do not merely ask it to “make this jazzy.”

For each sustained or metrically accented melody note, examine the chord active **during that note**, its duration, and where it resolves. Check both the sung melody and the instrumental solo. Short approach notes can support a tension that sounds harsh when held for two beats. Review the bass line, guide-tone motion and transitions into and out of any tonicization. Preserve successful colors rather than replacing every non-chord tone.

Examples of targeted repairs depend on context:

- A sustained A over `Db7` may clash with the chord's A-flat fifth. `Dbaug` retains the chromatic bass while including A; a prompt asking for an omitted fifth is less explicit than a suitable supported chord core.
- Sustained F-sharp over `Eb7` combines a minor third with the major third G. `Ebm7` can preserve the E-flat bass and directly support that melody tone.
- A tritone substitute can work as a brief approach but need earlier resolution when the next strong melody note arrives. Moving the chord boundary may preserve more color than replacing the whole bar.

These are musical options, not unconditional substitution rules. Recheck the surrounding progression, especially after changing one chord's third or fifth. Do not use a blanket “all melody notes must be chord tones” rule for jazz.

For a strict reharmonization:

```bash
python scripts/abc_tools.py inspect edited.abc --output edited-inspection.json
python scripts/abc_tools.py compare source.abc edited.abc --output melody-invariants.json
```

By default, `compare` requires the same quarter-note tempo. To permit an intentional tempo change while keeping pitches, relative note timing and bar meters:

```bash
python scripts/abc_tools.py compare source.abc edited.abc --allow-tempo-change
```

If melodic or metrical adaptation is authorized, a comparison failure can be expected. Record the precise changed passages and assess the still-fixed parts separately. This helper does not infer that a change is musically good or automatically waive differences. For a quoted theme, verify the complete intended phrase sequence and repetitions; a repeated opening motive is not the complete theme.

## Lyrics, syllables and phonemes

The generation API accepts lyrics and ABC, not a forced phoneme-to-note alignment channel. Keep alignment records in sidecar JSON/Markdown; do not add non-native phoneme annotations or `w:` fields to the model's ABC.

When changing language, adapt the lyrics for singing rather than translating word for word. Use the final score's resolved vocal notes to map every sung syllable to its intended note or contiguous melisma. Map consonants to attacks/releases and sustain vowels across melismas; one phoneme does not equal one note. Check lexical stress, short-note consonant density, long-note vowel choice, breathing rests, pickup syllables and section tags. If using a pronunciation dictionary, preserve the selected pronunciations and their provenance, and flag out-of-vocabulary words for manual review.

A useful sidecar records section, lyric line, word/syllable, phonemes, stress, selected voice, note indices, pitches and planned onset/duration. Validate that all intended vocal notes and syllables are accounted for. This proves a designed symbolic fit, not that the generated performance realizes it. Listen for dropped or repeated words, use ASR/PER as a separate pronunciation diagnostic, and use audio alignment if claiming measured synchronization.

## Native serialization, rendering and audio checks

The published [SheetSage2 notation module](https://huggingface.co/m-a-p/SheetSage2/blob/main/notation_sheetsage2.py) provides `score_to_abc(score)` and `validate_serialized_abc(text, score)`. The latter requires the original structured `RebuiltAbcScore`, including its event arrays; **it is not a free-text `validate_abc(text)` API**. The public remote-code model release is not an installable `sheetsage2_infer` parser package. When editing structured musical events inside that implementation, rebuild with its native serializer and retain its validation results. The portable helper here does not claim to reproduce that reconstruction check.

For an additional visual check, the downloaded SheetSage2 release can render an edited ABC without a model load:

```bash
python "$SHEETSAGE2_DIR/setup_render.py"
python "$SHEETSAGE2_DIR/render.py" --abc edited.abc --output rendered-edit --score pdf,svg
```

`SHEETSAGE2_DIR` is your downloaded public release directory. Rendering has optional dependencies installed by `setup_render.py`; none are required by this skill's ABC helper. The release's audio renderer uses supplied MIDI. Passing an old MIDI alongside edited ABC previews the old note timing/content, so regenerate corresponding MIDI before using it as evidence for an edit. Direct instrumental rendering does not test YuE2 audio adherence.

Finish with three distinct checks: native-dialect structure, the intended symbolic musical invariants, and the actual generated sound. Listen especially to the changed bars and their transitions. Save both source and edited ABC, prompts, lyrics, decoder identity, seeds, validation records and audio. Transcribing the result with SheetSage2 can provide a further diagnostic, but its transcription errors must not be mistaken for definitive generator errors.
