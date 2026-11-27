# YuE and ACE Studio Round Trip Test Plan

Establish whether we can develop a composition in either application, exchange it through MIDI and YuE-compatible ABC, edit it, and generate a new performance from the intended revision. Codex will perform the operations and record the evidence. The user supplies musical direction and reviews results when desired.

Created November 27, 2026. RT01 has been executed: musical notes and lyrics passed; raw MIDI lost key and section metadata; the workflow passed after explicit restoration. See the [RT01 results](yue-ace-rt01-results.md). RT02 through RT22 remain planned and not yet executed. Systematic experiments with YuE settings follow after the workflow baselines are established.

## Routes and lyric modes

**YuE to ACE and back:** YuE ABC plus a lyric sheet → MIDI plus lyric and section data → ACE project and audition → revised MIDI plus reconstructed lyrics → reviewed YuE ABC and tagged lyrics → generated take.

**ACE to YuE and back:** ACE composition and optional mapped lyrics → MIDI plus lyric and section data → reviewed YuE ABC and tagged lyrics → generated take → import that take's actual saved score into a separate ACE revision for comparison. Import the rendered audio separately when comparing what was played with what was written.

The return score is the symbolic plan supplied to or produced by YuE. It is not a transcription of the generated recording. Audio adherence must be assessed separately.

There are three distinct starting conditions: supplied lyrics; no supplied lyrics followed by a writing step; and deliberately empty lyrics for an instrumental. Studio's writing assistant or Codex can perform the writing step. Passing empty lyrics directly to generation requests an instrumental; it does not request automatic lyric authorship. Record who wrote each lyric draft and freeze the accepted text before generating. See the local generation reference at `docs/generation.md` and the [Studio workflow documentation](https://github.com/timoncool/YuE2-Studio).

## Fixtures and fixed conditions

Use original test material and separate test projects. Preserve the user's current ACE project and both applications' existing work.

- **Diagnostic fixture:** four to eight bars, C major, 4/4, a constant 96 BPM, a recognizable melody, and separate Vocal, Ins, and Chords roles. Include quarter, eighth, and sixteenth notes, a tie, a rest, and an offbeat entrance. These known events make conversion errors easy to identify.
- **Song fixture:** a short verse and chorus, normally 16 bars at 96 BPM, with an approved lyric sheet and an independently recorded section map. The chart's nominal duration is not a guarantee of the generated audio's length.
- **Instrumental fixture:** a short melody in Ins, with Vocal rests, chords, and explicitly empty lyrics.
- **Boundary fixtures:** small variations reserved for meter changes, pickup notes, split clips, extra tracks, and metadata loss. Introduce them only after the ordinary exchange works.

Record app and engine versions, model identities, writer configuration, source and destination files, MIDI timing resolution, import grid and track mapping, style, planning mode, all seeds, adapters, and the effective generation settings. For a baseline, use the installed defaults as captured at execution; do not silently substitute values from another implementation or an older document. Keep the baseline style, models, and settings fixed while testing a musical edit.

## Core route cases

| ID | Starting condition and route | Action | Required observation |
| --- | --- | --- | --- |
| RT01 | YuE with supplied lyrics → ACE → YuE | Compose or select a full score, transfer it with its lyrics, audition in ACE, and return it without musical edits | Notes, harmony, tempo, section order, and approved words match the source after conversion; the returned request uses that score |
| RT02 | YuE with an idea and no supplied lyrics → ACE → YuE | Have Studio's configured writing assistant draft lyrics, approve and freeze them, then perform RT01 | The writing step is explicit, its draft is saved, and its words survive the exchange |
| RT03 | YuE instrumental → ACE → YuE | Generate or select an instrumental score, audition it in ACE, and return it with empty lyrics | The melody stays in the intended instrumental role; assess the generated audio for unintended singing or vocal sounds |
| RT04 | ACE with supplied lyrics mapped to notes → YuE → ACE | Export the score and a separate lyric record, reconstruct tagged lines, generate, and return the saved score | Vocal and instrumental roles map correctly; words and sections correspond to the approved ACE composition |
| RT05 | ACE composition with no supplied lyrics; Studio writes | Supply the existing score as context to Studio's writer, fit the draft to the melody in ACE, freeze it, then perform RT04 | A usable lyric draft fits the existing phrases without silently replacing the approved composition; record any writer limitation |
| RT06 | ACE composition with no supplied lyrics; Codex writes | Codex writes original words against the note and phrase map, assigns them in ACE, then performs RT04 | The agent can bridge musical structure and tagged text while preserving the chosen melody; retain the exact alignment record |
| RT07 | ACE instrumental → YuE → ACE | Compose in ACE, convert MIDI to ABC, generate with empty lyrics, and return the saved score | Instrumental notes and harmony survive; assess the output for the intended theme and absence of unintended vocals |
| RT08 | ACE chord bed without a melody → YuE → ACE | Export supported chord material, reconstruct a chord-bed score, provide lyrics, and let YuE plan the melody | Planning adds the missing melody deliberately; save and return the newly produced score, not the original rests-only bed |

RT01 and RT04 first use the diagnostic fixture and stop before audio generation if the symbolic comparison fails. RT02 and RT05 exercise the configured Studio writer. If that writer is Codex through Agent MCP, record it as such; it is not independent evidence for another writing model. An unavailable writer produces a blocked result, not an unreported substitution.

## Musical edit cases

Start each case from an unchanged copy of a passing core fixture. Change only what the row authorizes.

| ID | Deliberate change in ACE | Required observation after return to YuE |
| --- | --- | --- |
| RT09 | Change two melody notes in one phrase | The intended pitch changes appear; other pitches, note timing, chords, and lyrics remain symbolically unchanged |
| RT10 | Replace one bar's harmony while retaining the melody | The new chord and its boundary survive; the melodic events remain unchanged; report any chord naming or inference |
| RT11 | Change only the constant tempo | Beat-relative notes and chords stay fixed, while the nominal time in seconds changes as expected; the request and style do not contain conflicting tempo instructions |
| RT12 | Move only the vocal part by one octave | Vocal pitches change by exactly 12 semitones; Ins and chord events remain fixed; record importer range adjustments explicitly |
| RT13 | Replace one lyric line and realign it in ACE | Reconstruct the approved wording in the correct section, check syllable and stress fit, and assess YuE's sung result without claiming forced note alignment |
| RT14 | Split words across notes and sustain a syllable over several notes | Reconstruct each word once, join its syllables correctly, and omit ACE continuation placeholders from the plain lyric sheet |

## Structure and boundary cases

| ID | Condition | Required observation |
| --- | --- | --- |
| RT15 | Verse, repeated chorus, and instrumental break | Section order, repetitions, lyric placement, and vocal rests survive; repeated text is not deduplicated or moved into the break |
| RT16 | Pickup, opening rest, and a vocal phrase split across clips | Local clip coordinates become the correct global onsets; neither the pickup nor intentional silence is trimmed or shifted |
| RT17 | A simple 3/4 composition | Meter and bar lengths survive the exchange; the importer does not silently impose 4/4 |
| RT18 | A small composition with tempo and meter changes | Determine which changes each format and converter actually supports; report any rejection, flattening, or approximation before generation |
| RT19 | Additional instrument tracks and polyphonic material | Make the reduction to YuE's Vocal and Ins lines explicit; preserve a dedicated chord source; report dropped notes and parts instead of claiming a complete arrangement transfer |
| RT20 | Missing or changed chord names, section markers, or lyric metadata | Detect the loss; restore from the independent section and lyric records where possible; label guessed harmony and revalidate it |
| RT21 | Three consecutive exchanges with no musical edits | Compare every revision with the original for accumulated timing drift, duplicate notes, octave changes, altered chords, and lyric duplication; defer synthesis until comparison is complete |
| RT22 | Save and reopen the test project and exchange files | Resume from recorded artifacts in a separate test project, verify the chosen revision and settings, and regenerate without reconstructing the composition from memory |

RT18 through RT20 probe known conversion boundaries. An unsupported capability is a useful finding, but it is not a successful preservation result. Restore the original fixture before testing another condition.

## Shared execution procedure

1. Read connection status and current project state. Use the confirmed local YuE MCP endpoint at `http://127.0.0.1:8791/mcp`; use ACE's preferred CLI when it can see the bridge, otherwise its working MCP connection. Read the exact schemas for the operations being used.
2. Work in an isolated test project and record a baseline manifest. Keep source ABC, lyrics, section map, and original MIDI immutable. Run heavy jobs sequentially.
3. Export or convert the composition. Map Vocal, Ins, and Chords explicitly. Account for MIDI tick resolution, the importer's quantization grid, and clip-local versus project timing.
4. Build and save the lyric transfer separately from MIDI. Read ACE's sentence and per-note data; reconstruct ordinary words and phrases; assign section tags from the recorded section map. ACE's automatic sentence breaks are timing-based and need review.
5. Preserve detailed syllable-to-note assignments in a sidecar file. Use them to check the text and musical fit; YuE does not accept them as an enforced alignment channel. Do not add unsupported lyric or phoneme directives to its ABC.
6. Apply only the case's intended change, export the new revision, convert it back, and validate its symbolic content. Record every unexpected alteration before starting generation.
7. Generate with the reviewed ABC and lyric sheet. For completed compositions, disable continuation. RT08 deliberately invokes melody planning over a chord bed and must retain its resulting plan.
8. Wait for completion, inspect the job's actual saved score and request, and save the audio. Assess the generated performance separately. Import the returned score into another ACE revision when the route requires it.
9. Record findings and artifact paths. If a prerequisite or observation is unavailable, mark the affected stage blocked or not assessed; do not infer success from a file existing or a launch being accepted.

## What counts as success

**Symbolic preservation:** compare interpreted musical events, not ABC text or MIDI bytes. Normalize onsets and durations to quarter-note fractions. Check pitches, merged tied durations, rests, note counts, chord identity and boundaries, meter, tempo, section order, and track-role mapping. The diagnostic fixture uses values representable on the selected grids, so unexpected rounding is a finding. For boundary material, record the measured approximation instead of inventing an acceptable tolerance afterward.

**Lyric preservation:** compare words, language, line order, section assignment, repeated sections, and the approved text. Save word reconstruction and melisma handling. An exact ACE assignment can be retained as evidence without claiming YuE consumed that assignment.

**Performance:** listen for the intended melody and harmony, phrase and section placement, lyric intelligibility, missing or repeated words, unwanted vocals in instrumentals, and unintended truncation. Record timestamped findings. If claiming measured note or word synchronization, use an appropriate audio analysis and identify its method; the returned symbolic score alone cannot prove it.

Report the symbolic, lyric, and performance outcomes separately as pass, fail, unsupported, blocked, not assessed, or not applicable. A successful transfer can coexist with a disappointing generated performance. A pleasant recording does not establish that the score survived correctly.

## Evidence retained for every case

Keep a case folder with the original and revised ABC, both MIDI exports, the tagged lyrics, the ACE project, the detailed lyric and section records, effective settings and seeds, tool/job identifiers, comparison results, generated audio, and listening notes. Use a new revision directory for every changed request. Record whether an artifact is an intended composition, a generated plan, a rendered recording, or an estimated transcription.

The case record contains: ID, fixture revision, date, route, lyric author, requested change, source and returned artifact paths, symbolic outcome, lyric outcome, performance outcome, observed losses, workarounds, and next action. All fields begin as planned or not assessed; the completed evidence determines the outcomes.

## Execution order and later YuE control experiments

First run the no-edit diagnostic portions of RT01 and RT04. Resolve conversion or lyric-transfer problems before judging generation. Next run RT03 and RT07, then the vocal and writing routes RT01, RT02, RT04, RT05, and RT06, followed by RT08. Run RT09 through RT14 as controlled musical changes. Finish with RT15 through RT22.

After a baseline passes, create separate experiments from its frozen artifacts. Change one named setting per experiment and retain the effective value, seed, and original comparison take. Candidate families are score planning mode and planning seed; harmony and section planning; semantic sampling and text guidance; synthesis steps and solver; and available model or adapter choices.

Separate settings that write a new composition from settings that perform an existing one. Studio documents harmony planning controls as ignored when a complete ABC is supplied. A fixed-score test therefore cannot establish whether those controls improve composition. Likewise, score-free generation needs its own comparison group, and a seed alone does not guarantee identical audio across changed requests or models.

## References

- The installed ACE command manuals for `import file`, `export midi`, `clip lyrics`, `clip note-content`, and project save operations, plus the [ACE Studio capability reference](ace-studio-reference.md).
- The installed YuE2 Studio schemas for `score_to_midi`, `score_from_midi`, `score_compose`, `score_chord_bed`, `assistant_write`, and `song_create`, and its [official workflow guide](https://github.com/timoncool/YuE2-Studio/blob/main/docs/mcp-skill.md).
- The [YuE generation reference](generation.md) and [ABC and lyric alignment reference](../skills/yue2-music/references/abc-editing.md).
