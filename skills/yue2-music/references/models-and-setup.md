# Models, setup, and the audio-to-score bridge

This reference targets the public YuE2 inference runtime **0.1.6**, the
SheetSage2 Transformers interface, and the MERT-v2 Transformers interfaces.
Download models from the public repositories below and record the actual model
and code revisions in each run manifest.

## What connects to what

| Model | Distribution ID | Role in this skill |
|---|---|---|
| YuE2-3B | `m-a-p/YuE2-3B` | Style and lyrics → optional symbolic plan → semantic tokens → acoustic latents. Used for generation and regeneration after editing. |
| YuE2-Vae | `m-a-p/YuE2-Vae` | Default listening decoder: acoustic latents → stereo audio. |
| YuE2-Vae-legacy | `m-a-p/YuE2-Vae-legacy` | Benchmark decoder. Decode the same latents with this model when reproducing the recorded evaluation protocol. |
| SheetSage2 | `m-a-p/SheetSage2` | Audio → melody, chords, beats, key, structure, ABC, and MIDI. Use it to obtain a cover/edit starting score or inspect generated audio. |
| MERT-v2-FullSong | `m-a-p/MERT-v2-FullSong` | SheetSage2's automatically loaded encoder parent. Also usable independently for continuous music features. |
| MERT-v2-30s | `m-a-p/MERT-v2-30s` | Optional continuous feature extractor for short recordings. Not required for generation, cover, or editing. |

The usual cover chain is:

```text
source audio
  → SheetSage2 (automatically loads its MERT-v2-FullSong parent)
  → melody_only=True → inspect/correct melody ABC without chord symbols
  → YuE2-3B with cot="melody", target style, and target lyrics
  → acoustic latents → YuE2-Vae → listening audio
```

For a score edit, keep or revise the chord symbols and use `cot="full"`.
For agentic editing, an agent edits the exported score and prompts between the
planning and regeneration steps. There is no additional "agentic" model API.

The public MERT-v2 encoders return **continuous features**, not the causal
discrete semantic token IDs used inside YuE2. Do not pass a MERT embedding to
YuE2 as semantic tokens. Plain YuE2 generation does not need a separate MERT
inference call. The two public MERT variants are bidirectional encoders; the
causal tokenizer lineage is a distinct model branch.

## Keep the environments separate

These releases pin different PyTorch, Transformers, and NumPy versions. Use
separate environments and exchange audio/ABC/MIDI files between them. A shared
Hugging Face cache is fine. Run the stages sequentially to release GPU memory
before loading the next model.

### YuE2

The release card targets Linux, Python 3.10+, and a 24 GB NVIDIA GPU with BF16
support. The package installs its own pinned dependencies. Do not substitute an
unverified package with a similar name from PyPI.

```bash
python3.12 -m venv .venv-yue2
.venv-yue2/bin/python -m pip install \
  git+https://github.com/multimodal-art-projection/YuE.git
```

Alternatively, install from a cloned official YuE repository with
`.venv-yue2/bin/python -m pip install /path/to/YuE`. The runtime pins PyTorch
2.10.0, Transformers 4.57.6, and NumPy 2.2.6. Current repository code and this
skill use Apache 2.0; earlier v0.1.6 wheel and skill ZIP archives retain their
bundled licenses. Model repositories supply weights independently of the runtime.
Use the generation examples in the main skill after installation.

### SheetSage2

There is no checked `pip install sheetsage2` distribution in this interface.
Download the model snapshot, install its requirements, and load it through
Transformers. Use Python 3.10 or 3.11. The model card specifies FFmpeg 6.1 and its
shared libraries; install these with the host's normal package/container tools
and confirm `ffmpeg` is on `PATH`.

```bash
python3.11 -m venv .venv-sheetsage2
.venv-sheetsage2/bin/python -m pip install huggingface-hub==0.36.0
.venv-sheetsage2/bin/huggingface-cli download m-a-p/SheetSage2 \
  --local-dir models/SheetSage2
.venv-sheetsage2/bin/python -m pip install \
  torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126
.venv-sheetsage2/bin/python -m pip install \
  -r models/SheetSage2/requirements.txt
```

Its requirements pin Transformers 4.45.2 and NumPy 1.24.3. Loading the adapter
snapshot automatically obtains the exact MERT-v2-FullSong parent selected by
its config, checks the parent files, and merges adapters in FP32. Do not replace
that parent with MERT-v2-30s or manually substitute another FullSong revision.
`trust_remote_code=True` executes the model repository's Python implementation;
use a reviewed revision and record it with the outputs.

## Transcription API

```python
from pathlib import Path
import torch
from transformers import AutoModel

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained(
    "models/SheetSage2", trust_remote_code=True,
).eval().to(device)

result = model.transcribe(
    "source.wav",
    output_dir="runs/source-score",
    dtype="bf16" if device == "cuda" else "fp32",
)
if result.get("abc_error") or not result.get("abc"):
    raise RuntimeError(f"No usable ABC: {result.get('abc_error')}")
print(result["warnings"])
print(Path("runs/source-score/score.abc"))
```

For direct Hub loading, replace the local directory with the repository ID and
pass the same recorded commit to `revision` and `code_revision`.

### One-step melody-only score for a cover

Use the current top-level option to keep both `Vocal` and `Ins` melodies while
omitting chord symbols from ABC and chord accompaniment from playback/combined
MIDI. The default full transcription is unchanged. This option changes exports,
not inference: raw predicted chord events and LAB annotations remain available
when the selected tasks include chord prediction.

```python
try:
    result = model.transcribe(
        "song.mp3", output_dir="cover-score", melody_only=True,
    )
except RuntimeError as error:
    partial = getattr(error, "result", None)  # Completed transcription, if available.
    if partial is not None:
        print(partial.get("abc_error"), partial.get("warnings", []))
    raise
abc = result["abc"]  # Also saved in cover-score/score.abc.
```

If the requested melody-only ABC cannot be built, Python raises `RuntimeError`
with the completed transcription attached as `error.result`; the CLI exits
nonzero. Do not treat saved annotations alone as a successful cover score.

```bash
.venv-sheetsage2/bin/python models/SheetSage2/infer.py song.mp3 \
  --output cover-score --melody-only
```

Review this ABC, then pass it to YuE2 with `cot="melody"` and the target style and
lyrics. The skill's `scripts/transcribe.py --task melody-full` enables
`melody_only=True` too; `--task melody-vocal` additionally selects only the vocal
melody task. Its `input.json` records the export flag and task prompts. These
two helper modes omit chord prediction tasks, so raw chord predictions are not
requested there. Use the direct default-task call above if retaining raw chord
annotations matters.

Older snapshots may lack this keyword. Refresh the downloaded model code to a
reviewed revision exposing `melody_only` explicitly; the helper refuses to
assume that an older `**kwargs` wrapper implements the export guarantee.

The callable interface is:

```text
model.transcribe(
    audio,
    output_dir=None,
    *,
    sampling_rate=None,
    dtype="bf16",                  # "bf16" or "fp32"
    preset="default",              # "default" or "paper"
    prompts=("timestamp", "downbeat_meter", "structure", "key",
             "chord_full", "melody_full"),
    max_seconds=None,
    overlap_seconds=None,
    lookahead_seconds=None,
    progress=None,
    export_logits=False,
    export_scores=False,
    export_embeddings=False,
    output_hidden_states=False,
    melody_only=False,              # Chord-free ABC and playback when True.
    render_audio=False,
    render_score=False,
    render_parts=("mix",),
)
```

- Paths, encoded audio bytes, and binary audio streams are decoded, mixed to
  mono, and resampled to 24 kHz automatically.
- NumPy arrays and tensors require `sampling_rate` and shape `[samples]` or
  **`[channels, samples]`**. `soundfile.read(..., always_2d=True)` instead returns
  `[samples, channels]`; transpose it before passing it to SheetSage2.
- The minimum accepted input is 1,025 finite samples after resampling. A short
  clip may still lack enough decoded beats or key information to construct ABC.
- The default uses 300-second windows, 200-second overlap, and 100-second
  lookahead. `max_seconds` deliberately crops the input; it is not a memory-only
  setting. Reducing overlap requires `0 <= lookahead <= overlap < 300`.
- `preset="paper"` fixes overlap to 100 seconds and lookahead to zero, uses the
  recorded audio frontend, and changes generation stopping. Use it only when
  reproducing that evaluation protocol, with its recorded task prompts.
- Optional logits and all-layer exports can be large. Leave them disabled for
  ordinary cover/edit workflows. The model reports `peak_gpu_mib` and window
  statistics; use these observations when sizing a particular workload.

### Choosing transcription tasks

Task names are not free-form prompts. `chord_full` and `chord_majmin` are mutually
exclusive, as are `melody_full` and `melody_vocal`. Timed export requires
`timestamp`; usable ABC also needs decoded beats and key information.

For a source vocal melody without chord conditioning:

```python
result = model.transcribe(
    "source.wav", output_dir="runs/source-melody",
    prompts=("timestamp", "downbeat_meter", "structure", "key", "melody_vocal"),
    melody_only=True,
)
```

Use `melody_full` instead to retain both vocal and instrumental melody tracks.
For harmony-aware editing, use the default six tasks. After every transcription,
inspect `warnings`, `diagnostics`, `abc_error`, and the score before regeneration.
Transcription can contain musical errors even when its notation is valid.

### Outputs

| Data | In-memory value | File output |
|---|---|---|
| ABC score | `result["abc"]`: string; default full mode can return `None` on notation failure | `score.abc` |
| Combined playback; chords omitted with `melody_only=True` | `result["midi"]`: bytes | `transcription.mid` |
| Melody tracks | `result["midis"]["melody"]`, `melody_vocal`, `melody_instrumental`: bytes | `melody.mid`, `melody_vocal.mid`, `melody_instrumental.mid` |
| Chord playback; no chord notes with `melody_only=True` | `result["midis"]["chords"]`: bytes | `chords.mid` |
| Timed events | `result["events"]`: list; `result["num_events"]`: count | `events.json` |
| Annotation text | `result["labs"]`: mapping | `beat.lab`, `downbeat.lab`, `key.lab`, `chord.lab`, `structure.lab`, melody LABs |
| Per-window token IDs | `result["tokens"]`: list | `tokens.json`, `tokens.txt` |
| Optional tensors | `result["tensors"]`: CPU tensors grouped by window | `tensors/`, when requested with an output directory |

`output_dir=None` keeps inference results in memory. The saved `result.json`
uses an integer `events` statistic, whereas the Python result uses an event
list; do not confuse these two schemas. The `notation/` companions contain the
beat grid, intervals, and monophonic MIDI used for score reconstruction. Raw
MIDI retains timing details that the quantized notation can simplify.

## Canonical notation and rendering

SheetSage2's exported ABC uses its native two-voice serializer and validates the
result against the reconstructed score. The bundled module exposes:

```text
generate_abc_from_exports(melody_midi_path, *, output_path=None,
                          meter_conflict="infer", melody_only=False)
# -> (abc_text, score_object, companion_paths)

generate_abc_from_data(melody_midi, beats, chords, keys, structures, *,
                      meter_conflict="infer", melody_only=False)
# -> (abc_text, score_object)

score_to_abc(score_object)              # Also validates its own serialization.
validate_serialized_abc(text, score_object)
```

These are functions in the downloaded implementation, not a general ABC parser
or a one-argument `validate_abc(text)` API. New transcriptions can use the
top-level `melody_only=True` option. For an existing full transcription, the
following remains useful to reserialize its saved melody without chord symbols
and without another model inference call:

```python
from importlib import import_module
from pathlib import Path

package = model.__class__.__module__.rsplit(".", 1)[0]
notation = import_module(package + ".notation_sheetsage2")
abc, score, _ = notation.generate_abc_from_exports(
    "runs/source-score/notation/song_melody.mid", melody_only=True,
)
Path("runs/source-score/melody-only.abc").write_text(abc, encoding="utf-8")
```

The file helper requires the exact `*_melody.mid` name and sibling `*_beats.txt`,
`*_keys.txt`, and `*_structures.txt`; full-score conversion also needs
`*_chords.txt`. It does **not** infer a beat grid or key from an arbitrary MIDI.
Use the skill's ABC checks for agent-authored text, including duration, voice,
pitch, and edit-preservation checks.

Rendering is optional and does not use a YuE2 VAE:

```bash
.venv-sheetsage2/bin/python models/SheetSage2/setup_render.py
.venv-sheetsage2/bin/python models/SheetSage2/infer.py source.wav \
  --output runs/source-score --render-audio --render-score pdf,svg,png
.venv-sheetsage2/bin/python models/SheetSage2/render.py \
  --input runs/source-score --output runs/source-rendered \
  --audio --score pdf,svg,png
```

On minimal Linux installations the renderer also offers `setup_render.py
--with-deps`. It installs browser/rendering dependencies. The piano preview
uses MIDI note timing; sheet rendering uses ABC. **Editing only `score.abc`
does not update the existing MIDI**, so a subsequent piano preview can still
play the old notes. Synchronize MIDI after score edits with a compatible ABC
converter before judging the edited composition by its piano audio.

For self-contained offline model use:

```python
model.save_pretrained("models/SheetSage2-merged")
offline_model = AutoModel.from_pretrained(
    "models/SheetSage2-merged", trust_remote_code=True, local_files_only=True,
).eval().to(device)
```

Load/download the required files once before going offline. A raw adapter-only
snapshot still needs its parent files; the merged save removes that dependency.

## Optional MERT representations

MERT feature extraction is useful for a separate retrieval or analysis tool.
It is not needed to connect SheetSage2 to YuE2, and an embedding distance alone
does not establish melodic or harmonic fidelity.

```bash
python3.11 -m venv .venv-mert2
.venv-mert2/bin/python -m pip install \
  torch==2.6.0 torchaudio==2.6.0 transformers==4.53.2 \
  huggingface-hub safetensors soundfile
```

```python
import soundfile as sf
import torch
import torchaudio.functional as AF
from transformers import AutoFeatureExtractor, AutoModel

repo = "m-a-p/MERT-v2-FullSong"  # Or m-a-p/MERT-v2-30s.
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoFeatureExtractor.from_pretrained(repo, trust_remote_code=True)
encoder = AutoModel.from_pretrained(repo, trust_remote_code=True).eval().to(device)
audio, rate = sf.read("source.wav", dtype="float32", always_2d=True)
waveform = torch.from_numpy(audio[:30 * rate].mean(axis=1))
waveform = AF.resample(waveform, rate, processor.sampling_rate)
inputs = processor(waveform.numpy(), sampling_rate=processor.sampling_rate,
                   return_tensors="pt").to(device)
with torch.inference_mode():
    output = encoder(**inputs, output_hidden_states=True)
mask = output.feature_attention_mask[..., None]
embedding = (output.last_hidden_state * mask).sum(1) / mask.sum(1).clamp_min(1)
```

Both take 24 kHz mono and return 25 Hz, 1,024-dimensional frame features.
`hidden_states` has exactly 24 post-block tensors: index 0 is block 1, not an
input embedding. FullSong is adapted to full songs of 30–360 seconds; the
example deliberately uses only 30 seconds. Remove that slice for full-song
analysis within the intended context, and chunk longer recordings explicitly.
The low-level model does not provide the SheetSage2 whole-song stitching API.

## Distribution and license boundaries

The checked model cards identify the model weights as **CC BY-NC 4.0**. The
license of this skill does not relicense those weights or remove their
noncommercial terms. Code and dependencies retain their own applicable terms.
Link users to each model's `LICENSE` and `THIRD_PARTY_NOTICES.md`; do not bundle
model weights, authentication material, cached datasets, or unrelated examples
into the skill archive.

Rendering dependencies also have separate terms: abcjs is MIT, Playwright is
Apache 2.0, Chromium ships its own notices, and the bundled FluidR3 piano
samples carry attribution under CC BY 3.0 US. The skill can call installed
renderers without redistributing these assets.

Primary distribution documentation:

- [YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B)
- [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae)
- [YuE2-Vae-legacy](https://huggingface.co/m-a-p/YuE2-Vae-legacy)
- [SheetSage2](https://huggingface.co/m-a-p/SheetSage2)
- [MERT-v2-FullSong](https://huggingface.co/m-a-p/MERT-v2-FullSong)
- [MERT-v2-30s](https://huggingface.co/m-a-p/MERT-v2-30s)
