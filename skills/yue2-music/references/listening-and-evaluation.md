# Listening delivery and reproducible evaluation

Finish with audio the user can play and the exact conditions that produced it. A score, a metric or a successful process exit is not an audible result.

## Keep listening and scoring versions separate

Use `m-a-p/YuE2-Vae` for native listening previews. Use `m-a-p/YuE2-Vae-legacy` when reproducing the supplied benchmark-decoder protocol. Keep full model names, revisions and hashes in the record; the word “legacy” does not establish which decoder a score used.

Generate the acoustic latents once for a given song and decode that same saved `latent.npy` for each requested decoder. A musical or lyric edit requires new generation; changing only the decoder does not. The helper produces a new native result directory for cached decoding:

```bash
python scripts/run_yue2.py decode --source outputs/jazz \
  --output outputs/jazz-evaluation \
  --model "$YUE2_MODEL_DIR" --vae "$YUE2_EVAL_VAE_DIR" --offline
```

Use a verified local model matching the source result. For Hub models, pass the verified `--revision` and `--vae-revision` as well. Keep the source untouched. The new result must retain the original request, exact plan/semantic tokens and latents, with the new decoder's identity, configuration and audio hashes. Record the source latent hash and the decode-only operation; do not report decoder runtime as full generation speed.

A bare FLAC plus a custom decoder manifest is useful for listening, but is not sufficient input to the complete kit's frozen evaluator. Its adapter verifies a full native `SongResult.save_artifacts` directory, including `result.json`, request/config, token arrays, latents and audio hashes. Never copy an old manifest over new audio or keep the listening decoder's identity on the evaluation audio.

## Build the listening comparison

```bash
python scripts/listen.py outputs/pop outputs/jazz --output outputs/comparison
```

The helper creates a local HTML listening page with copied audio and exact requests. It does not upload or publish. Use fresh output directories and deliver the generated page alongside direct audio links when the interface supports playback.

Include the full songs and useful excerpts for long edits. Start excerpts before the edited passage and include its exit; a few isolated notes hide transition problems. For a vocal rewrite, include enough verse and chorus to assess words and phrasing. For a theme/solo edit, include both complete theme statements and their continuation, not just the opening motive.

Each listening entry should identify:

- Version and intended change, full style prompt, full lyrics and source/edited ABC.
- Actual model and decoder identity, seed, mode and any parameter overrides.
- Duration, truncation flags, structural/invariant checks and relevant change records.
- The complete audio, any excerpt's start/end times and whether it was normalized or otherwise processed.

Retain lossless generated audio even if making MP3 previews. Never replace or normalize the audio underlying recorded scores when refreshing listening links. A listening-page conversion is a delivery artifact; the frozen evaluation adapter applies its own recorded preprocessing.

Listen specifically for melody realization, chord clashes at sustained notes, instrument choices, lyric omissions/repetitions, pacing, section transitions and the ending. If no audio audition capability was available, say which checks were actually performed rather than claiming to have heard the song.

## Separate the evidence

| Check | What it supports | What it does not establish |
|---|---|---|
| Native ABC inspection | Accepted structure, time grid and supported symbols | Pleasant harmony or audible adherence |
| Sounding-note comparison | Specified symbolic pitches/onsets/durations preserved | Identical generated performance or waveform |
| SheetSage2 transcription of generated audio | A diagnostic estimate of realized musical events | Error-free ground truth |
| ASR and phoneme error rate | Recognized lyric/phoneme agreement under that protocol | Measured syllable-to-note synchronization |
| Audio forced alignment | Estimated word/phoneme timing, when checked | Perfect melody or arrangement quality |
| SongBench and other quality/control metrics | Their named metric under the recorded evaluator | Proof of a specific instrument, jazz authenticity or universal quality |
| Listening | The reported audible observations | An automatic benchmark result or an unperformed preference study |

Keep a designed lyric–phoneme–note sidecar separate from measured audio alignment. For claims of synchronized pronunciation, retain the aligner's output and review difficult words, melismas and instrumental sections. A lower PER alone does not prove correct note timing.

## Use a separate, complete benchmark package

The public YuE2 runtime and this skill do not distribute the complete scoring
code, evaluator weights or benchmark inputs. They do not expose `yue2 eval`,
`bench` or `verify` commands. When a separate evaluation package is available,
follow that package's documented entrypoints and frozen asset manifest.

Require the package's exact dataset/split definitions, preprocessing, decoder
identity, scorer revisions and hashes before reproducing a published result.
Use benchmark-decoded native result directories from `SongResult.save_artifacts`;
keep the reference lyric language and all attempted modes in the input record.
A prepare-only validation is not a measured score.

If scoring assets are unavailable, deliver the listening and symbolic checks
and report evaluation as unavailable. Do not substitute a similarly named
metric or fabricate a score. Evaluator GPU requirements are separate from the
24 GB YuE2 generation baseline.

## Prepare public listening artifacts deliberately

The listening helper creates a local bundle. It withholds credential-like
metadata and excludes weights and latents, but exact requests, local model
paths and failure messages can still be present in copied files. Review the
bundle before sharing it publicly. Share only the intended audio, scores,
prompts, lyrics and public model identifiers; omit private paths, account
identifiers, raw logs and unrelated sidecars.

Keep the original native result directory unchanged. If preparing a sanitized
public metadata export, give it its own manifest and hashes; do not present
modified metadata as the original generation receipt. Do not alter or replace
the preserved benchmark audio when updating listening previews.

## Report actual outcomes and preserve failed attempts

Retain the expected request list before running anything. For each requested mode/version, record success or failure, the failure reason, both truncation flags, decoder identity and every completed metric. Report sample counts and denominators. Keep partial/missing scores visible; do not silently discard an unsuccessful mode or choose a favorable seed after seeing the scores.

For SongBench, retain its seven dimensions and the reported global average: Melody, Arrangement, Musicality, Vocal, Instrumental, Mixing and Structure. Identify the exact version and decoder associated with each result. Do not reinterpret the global average as a dedicated jazz or harmony-consistency score.

The checked PER protocol runs four ASR passes. Preserve their transcripts, selected result and first-pass result; state the protocol when reporting the score. Do not replace it with a single transcription while retaining the same protocol label. For revised English lyrics, check the reference language and actual new lyric text, and inspect residual errors rather than treating the scalar as complete verification.

Report small sanity checks as small sanity checks. An average from a few self-written prompts is not the full benchmark, and a personal edit comparison is not an independent quality ranking. `complete:true` in an evaluation summary means the requested metrics were produced; it does not by itself mean quality acceptance passed. Keep model generation, benchmark measurement and musical judgment traceable as separate claims.
