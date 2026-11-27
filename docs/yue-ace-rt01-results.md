# RT01 YuE and ACE Round Trip Results

Completed November 27, 2026. The workflow **passes with explicit lyric and metadata preservation**. The raw MIDI exchange preserves the musical notes, harmony, tempo, and meter, but loses the key signature and section markers. Those losses were recorded and restored from the source before generation.

The user heard the ACE audition and the final YuE take. The ACE melody was judged extremely high but otherwise good; the generated YuE tune and lyrics were judged correct. This is a successful functional exchange with a documented limitation, rather than evidence that plain MIDI carries the complete composition package.

## Results

| Check | Result | Evidence |
| --- | --- | --- |
| Vocal notes | Pass | All 56 pitches, onsets, and durations match |
| Instrumental melody | Pass | All 54 pitches, onsets, and durations match |
| Chord notes | Pass | All 179 pitches, onsets, and durations match |
| Harmonic progression | Pass | All 30 entries after collapsing consecutive repeated chord symbols match |
| Tempo and meter | Pass | 95 BPM and 2/4 preserved |
| Raw MIDI key signature | Lost | YuE inferred E-flat instead of reading a stored key signature |
| Raw MIDI section markers | Lost | Intro, verse, chorus, and outro collapsed to one generic verse on reconstruction |
| Restored metadata | Pass | One key event and four section events restored; no note changed |
| Lyric reconstruction | Pass | Eight lines and 46 words read back from ACE match the supplied lyric sheet |
| Generation input | Pass | The saved generation record contains the returned ABC and reconstructed lyrics |
| Generated audio | Pass by user listening | User reported that the tune and lyrics sound correct |

## Test composition

The original test lyric was written by Codex and supplied before composition. YuE composed in E-flat, 2/4, at 95 BPM, with 48 bars and a nominal chart duration of 60.632 seconds. The style brief requested 96 BPM; the effective style was reconciled to the actual 95 BPM chart before generation. The planner's choices were retained as the baseline without changing notes or chords.

The vocal spans MIDI pitches 70 through 84, B-flat4 through C6. The high range was present in the original YuE score and remained unchanged during transfer. It explains the user's ACE listening observation; it is not an accidental octave shift introduced by the exchange. A lower-range version should be a separate musical-edit test.

## Procedure completed

1. Used the working local YuE MCP connection and composed one score from the supplied verse and chorus, with planning seed 424201.
2. Exported the source ABC to MIDI and imported it into a separate saved ACE project, adopting its tempo and meter.
3. Restored the Vocal, Ins, and Chords track names from the imported clip identities. ACE initially gave all three tracks a generic MIDI track name.
4. Loaded Elirah on the vocal track and assigned all eight lyric lines. Dry runs and committed fills reported no discarded text. The note pitches and timing remained unchanged.
5. Auditioned in ACE. The user reported that the melody was extremely high but otherwise sounded good.
6. Exported MIDI both before and after the vocal conversion and lyric assignment. An independent MIDI event comparison found all 289 note events unchanged in both exports.
7. Reconstructed YuE ABC from the source MIDI and the ACE export. Both preserved the melody and harmony; the ACE export lacked key and section metadata.
8. Added only the missing source key and marker events to a separate preserved MIDI file. Compared its note events again, then reconstructed and validated the final return ABC.
9. Rebuilt the lyric sheet from ACE's lyric readback using the retained line-to-note and section mapping. ACE's timing-based sentence splitter separated the last line into two sentences; the recorded line boundaries restored it correctly.
10. Generated the YuE take with the returned score, continuation off, full score guidance, audio seed 424202, 32 midpoint steps, no additional LoRA, and lossless FLAC output. Confirmed the saved score and lyrics against the actual request.
11. Played the result in YuE. The user reported: “Tune and lyrics sound correct.” Paused the completed test playback and restored the player's original repeat setting.

## Generated recording and verification limits

The generated FLAC is stereo at 48 kHz and decodes to 2,835,776 sample frames, or 59.078667 seconds. Its measured peak is approximately 0.698, with no decoded samples at or above full scale. The recording is shorter than the nominal chart duration by about 1.553 seconds; musical score timing is not a sample-exact waveform contract.

The returned symbolic score is the composition plan, not a transcription of the generated recording. Note-by-note audio alignment was not measured. Performance acceptance here comes from the user's listening checks, supported by file decoding and verification of the actual generation input.

ACE was no longer running when an optional audition WAV export was attempted after the successful round trip. The cause is unknown. The ACE project and both MIDI exports had already been saved; the optional ACE audition audio was not archived. No RT02 or later test was run.

## Artifacts and reproduction

The external run archive is identified as `RT01/run-20261127-001/`. The table lists paths relative to that archive. Audio, the ACE project, and raw runtime responses are local artifacts and are not included in this repository; the measurements and listening observations above summarize that run.

| Artifact | File |
| --- | --- |
| Saved ACE project | `RT01-ACE/RT01-ACE.acep` |
| Original composition and MIDI | `source.abc`, `source.mid` |
| ACE raw export | `ace-return.mid` |
| Return MIDI with restored metadata | `ace-return-preserved.mid` |
| Final return score | `returned.abc` |
| Actual generated plan | `generated-plan.abc` |
| Original and returned lyric sheets | `lyrics-original.txt`, `lyrics-returned.txt` |
| Lyric line mapping and ACE readback | `lyric-line-map.json`, `ace-lyrics-readback.json` |
| MIDI and symbolic comparisons | `midi-comparison.json`, `symbolic-validation.json` |
| Generation and audio verification | `generation-validation.json`, `audio-validation.json` |
| Effective generation settings | `generation-effective.json` |
| Recording | `RT01-returned.flac` |
| Complete run summary | `manifest.json` |
| MIDI inspection and restoration helper | `midi_roundtrip.py` |

YuE library title: **RT01 Morning Light - ACE round trip**. Song ID: `01a21dc8-20e4-759e-80a6-a27559261ca0`. Song job: `67da7fc4fcaf46e9`. The engine-default snapshot and full tool responses are retained in the same folder.

## Consequence for the remaining tests

Use a composition package containing MIDI, tagged lyrics, section boundaries, key, and the note-to-lyric record. Preserve and compare raw exports before applying any repair. Restore metadata only when it remains valid for the intended revision; later key or section edits must update the companion record instead of copying stale source metadata.

Future tests should explicitly distinguish transfer changes from deliberate musical changes. In particular, the high vocal register is a good RT12 octave-edit candidate, while RT04 exercises composing in ACE first. The remaining 21 cases stay planned.
