# Generation, covers and reusable stages

Use the `yue2-infer` 0.1.6 interface described here. Follow [models-and-setup.md](models-and-setup.md) for installation and public model snapshots. Record the package version, model revisions and weight hashes with each run. Validate a new installation with fresh outputs before treating it as a reproduced experiment.

## Choose the conditioning mode

| Task | `cot` | ABC input |
|---|---|---|
| Generate melody, harmony and audio from text | `full` | Omit; YuE2 plans a melody with chords |
| Generate a melody and then audio | `melody` | Omit; YuE2 plans without chord symbols |
| Generate audio directly from text | `off` | Must be omitted |
| Cover an existing melody in another style | `melody` | Supply melody-only ABC with chord symbols removed |
| Reharmonize or otherwise edit a score | `full` | Supply the edited melody-and-chord ABC |

These modes select different native instructions. Do not paste your own replacement instruction into the style prompt. An external ABC input bypasses the symbolic planner; it does not call a second planner to repair the score. `cot="melody"` does not remove chords from an input file automatically.

YuE2's song request accepts `style`, `lyrics`, `cot`, `seed`, `abc`, `cfg_scale` and `id`. `tags` is an alias for `style`; the two must agree if both are supplied. There is no request field for `reference_audio`, `phonemes`, `bpm`, `negative_prompt`, an edit interval or a reference singer. Put tempo and meter in the ABC and describe them consistently in the style. Keep pronunciation and note-alignment instructions in validation sidecars; they are not hard conditioning inputs.

## Generate and save all three modes

The following example uses local, verified snapshots; set the directory variables to the snapshots obtained during setup. Keep the full sectioned lyrics in `lyrics.txt`. Run each pipeline sequentially.

```python
import os
from pathlib import Path
from yue2 import YuE2Pipeline

model = os.environ["YUE2_MODEL_DIR"]
listening_vae = os.environ["YUE2_LISTEN_VAE_DIR"]
style = (
    "English, warm female vocal, contemporary pop, 96 BPM, "
    "piano, rounded electric bass, restrained drums, clear diction"
)
lyrics = Path("lyrics.txt").read_bytes().decode("utf-8")

with YuE2Pipeline.from_pretrained(
    model, vae=listening_vae, device="cuda", local_files_only=True,
) as pipe:
    for mode in ("full", "melody", "off"):
        output = Path("outputs") / f"original_{mode}"
        output.mkdir(parents=True, exist_ok=False)
        song = pipe(
            id=f"original_{mode}", style=style, lyrics=lyrics,
            cot=mode, seed=831001,
        )
        receipt = song.save_artifacts(output)
        print(mode, receipt["status"], receipt["truncated"])
```

For a Hub load, use `from_pretrained(repo, revision=model_revision, vae=vae_repo, vae_revision=vae_revision, device="cuda")` with independently verified revisions. `cache_dir`, `token` and `local_files_only` are supported. Keep tokens in the authentication environment, out of scripts and manifests.

`save_artifacts` saves `audio.flac`, `score.abc` when applicable, exact ABC/prefix IDs, semantic tokens, `latent.npy`, the request, configuration, timing, decoder identity and hashes. `song.save("song.flac")` or `.save("song.wav")` saves only audio. MP3 is a separate delivery conversion. The Python save methods can overwrite existing files: require a fresh destination as above.

`status="complete"` does not imply an untruncated song. Inspect both ABC and semantic truncation flags, then check duration, the ending, audibility and the requested musical behavior.

## Export, edit and regenerate

Generate the full plan once. Keep its directory immutable and edit a copy of `score.abc`. Validate the edited score before synthesis; see [abc-editing.md](abc-editing.md).

```python
from pathlib import Path
from yue2 import YuE2Pipeline

with YuE2Pipeline.from_pretrained(
    model, vae=listening_vae, device="cuda", local_files_only=True,
) as pipe:
    original_dir = Path("outputs/original_plan")
    original_dir.mkdir(parents=True, exist_ok=False)
    plan = pipe.plan(
        id="original", style=style, lyrics=lyrics,
        cot="full", seed=831001,
    )
    plan.save(original_dir)

# Copy the score to edits/jazz.abc; edit and validate it in a separate step.
```

Once the edited file exists:

```python
jazz_style = (
    "English, intimate female jazz vocal, relaxed 88 BPM, "
    "piano, tenor saxophone, upright bass, brushed drums, "
    "no guitar, spacious modern harmony, natural English phrasing"
)
edited_abc = Path("edits/jazz.abc").read_bytes().decode("utf-8")
output = Path("outputs/jazz_edit")
output.mkdir(parents=True, exist_ok=False)
with YuE2Pipeline.from_pretrained(
    model, vae=listening_vae, device="cuda", local_files_only=True,
) as pipe:
    song = pipe(
        id="jazz_edit", style=jazz_style, lyrics=lyrics,
        cot="full", seed=831001, abc=edited_abc,
    )
    song.save_artifacts(output)
```

This regenerates the song from the revised musical conditions. It does not promise unchanged waveforms, singer identity or performance outside an edited passage. A fixed seed makes comparisons easier to trace; different conditions still produce different audio.

For agentic editing, pass the exported score, original request and explicit invariants to a music-editing agent. Request an edited ABC file and a concise bar-by-bar change record. Validate melody preservation where requested, chord timing and compatibility, meter, ties, phrase lengths and lyric syllables before rendering. An English-lyric change often needs stress and vowel-duration adjustments even when its syllable count matches the old lyrics.

## Cover from a score or a recording

For a score-based cover, use a validated melody-only ABC and the desired lyrics:

```python
melody_abc = Path("edits/source_melody.abc").read_bytes().decode("utf-8")
with YuE2Pipeline.from_pretrained(
    model, vae=listening_vae, device="cuda", local_files_only=True,
) as pipe:
    output = Path("outputs/cover")
    output.mkdir(parents=True, exist_ok=False)
    song = pipe(
        id="cover", style=jazz_style, lyrics=lyrics,
        cot="melody", seed=831001, abc=melody_abc,
    )
    song.save_artifacts(output)
```

For an audio-based cover, transcribe the source recording with SheetSage2, review its melody and timing, remove the chord annotations, then use the same call. Obtain or check the lyrics separately. See [models-and-setup.md](models-and-setup.md) for the transcription model chain. Source-separation, transcription, lyric recognition and score-conditioned generation are distinct operations; YuE2 has no direct audio-upload argument in this interface. Its VAE encoder is not a substitute for melody transcription.

## Preserve exact stage outputs

The supported stage sequence is:

```python
from yue2 import SymbolicPlan

plan = SymbolicPlan.load("outputs/original_plan")
with YuE2Pipeline.from_pretrained(
    model, vae=listening_vae, device="cuda", local_files_only=True,
) as pipe:
    semantic = pipe.generate_semantic(plan)
    latents = pipe.synthesize(semantic)
    audio = pipe.decode(latents)
```

This continues the original plan without decoding and retokenizing its ABC. `SymbolicPlan.load` verifies the saved manifest and token arrays; modifying a file inside that saved plan intentionally makes validation fail. Supply revised ABC through a new request instead.

There is no `SemanticResult.load`, `SongResult.load` or general automatic partial-resume method. Stage calls return objects/arrays; save the plan, exact semantic tokens, latent array and provenance explicitly if constructing your own staged runner. Editing style, lyrics or ABC requires new semantic generation and synthesis. Cached latents are reusable when changing only the decoder, not when changing the music.

## Decode the same latents for listening and evaluation

Use `YuE2-Vae` for the listening/default version and `YuE2-Vae-legacy` when reproducing the documented benchmark decoder. Record the full model name and hash rather than inferring chronology from “legacy.” Keep the two audio files and their manifests separate.

```python
import hashlib
import json
import numpy as np
import soundfile as sf

evaluation_vae = os.environ["YUE2_EVAL_VAE_DIR"]
source = Path("outputs/jazz_edit/latent.npy")
latents = np.load(source, allow_pickle=False)
output = Path("evaluation/jazz_edit")
output.mkdir(parents=True, exist_ok=False)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

with YuE2Pipeline.from_pretrained(
    model, vae=evaluation_vae, device="cuda", local_files_only=True,
) as decoder:
    audio = decoder.decode(latents)
    sf.write(output / "audio.flac", audio, 48000, subtype="PCM_24")
    manifest = {
        "operation": "decode_cached_latents",
        "decoder_role": "evaluation",
        "weights": decoder.weights,
        "latent_sha256": sha256(source),
        "audio_sha256": sha256(output / "audio.flac"),
        "sample_rate": 48000,
        "decoder_dtype": "float32",
        "core_frames": decoder.vae_core_frames,
        "halo_frames": 16,
    }
    (output / "decoder.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
    )
```

This loads the decoder without generating a new song. Accepted latent shapes are `[T,64]` and `[1,64,T]`; the returned array is samples × two channels. Do not put this new audio under the old `SongResult` manifest, whose decoder identity describes the listening version. `decode(..., vae=...)` also exists, but it accepts no separate revision argument; use a verified local snapshot or configure the second pipeline as above.

## CLI equivalents and resume behavior

Given request JSON containing `id`, `style`, `lyrics` and optionally `seed`:

```bash
yue2 generate --request requests/original.json --cot full --output outputs/full
yue2 generate --request requests/original.json --cot melody --output outputs/melody
yue2 generate --request requests/original.json --cot off --output outputs/off
yue2 generate --request requests/original.json --stage plan --output outputs/plan
yue2 generate --request requests/jazz.json --cot full --abc-file edits/jazz.abc --output outputs/edited
yue2 generate --request requests/cover.json --cot melody --abc-file edits/source_melody.abc --output outputs/cover
yue2 batch --input requests/all.jsonl --output outputs/batch
```

Pass `--model` and `--vae` with the verified local snapshots and `--offline` for an offline run. For Hub snapshots, add `--revision` and `--vae-revision`. CLI output is nested under `--output/<request-id>/`. JSON may use `abc_path` relative to the request file; the Python API takes `abc` text. Batch requests require unique IDs and run sequentially.

`--resume` validates and reuses a **completed** matching result, including its hashes; it does not continue interrupted AR/NAR generation. Use a fresh directory after an incomplete run or when any request, model or configuration changes. CLI stages are `plan` and `audio`, not separate semantic/synthesis/decode subcommands.

## CFG, defaults and evaluation scope

Semantic CFG defaults to 1.0 in `full`/`melody` and 1.01 in `off`. For an explicit experiment, pass `cfg_scale=1.2` or CLI `--cfg-scale 1.2`. It is not a guaranteed quality improvement. With a score, the negative branch retains the same instruction and exact ABC while removing style and lyrics. ABC generation itself has no CFG.

The standard preset uses BF16 AR/NAR, FP32 VAE, 32 midpoint synthesis steps and context 24576. Preserve it for reproducible comparisons. The supported baseline is a BF16-capable NVIDIA GPU with 24GB memory and one active request per pipeline. Optional backend or precision changes need their own validation; do not silently shorten the song or reduce synthesis steps to report a successful baseline run.

The public runtime supplies `yue2 doctor`, `generate`, `batch`, and the Python API. Frozen benchmark evaluation requires separate scoring code and assets; see [listening-and-evaluation.md](listening-and-evaluation.md). SongBench scores, ASR/PER, score checks and listening answer different questions. Report actual checks and retain failures, truncation flags and every requested mode; none alone proves exact score or phoneme adherence in the audio.
