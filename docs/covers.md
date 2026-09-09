# Cover a recording

A cover starts with a readable composition: transcribe the source recording, review the melody, then ask YuE2 to realize it in a new style. The general YuE2 checkpoint supports this workflow without cover-specific fine-tuning.

## 1. Set up SheetSage2 separately

SheetSage2 and YuE2 use different dependency versions. Keep separate environments and exchange ABC and audio files. Run transcription and generation sequentially so they can share one GPU.

SheetSage2 uses Python 3.10 or 3.11 and requires FFmpeg 6.1 and its shared libraries. Follow the platform setup in its [model card](https://huggingface.co/m-a-p/SheetSage2), then run from the YuE repository root:

```bash
python3.11 -m venv .venv-sheetsage2
.venv-sheetsage2/bin/python -m pip install huggingface-hub==0.36.0
.venv-sheetsage2/bin/huggingface-cli download m-a-p/SheetSage2 \
  --local-dir models/SheetSage2
.venv-sheetsage2/bin/python -m pip install \
  torch==2.8.0 torchaudio==2.8.0 \
  --index-url https://download.pytorch.org/whl/cu126
.venv-sheetsage2/bin/python -m pip install -r models/SheetSage2/requirements.txt
```

Loading SheetSage2 automatically loads the MERT-v2-FullSong encoder selected by its configuration. A separate MERT2 feature-extraction step is unnecessary.

## 2. Transcribe and review the melody

```bash
.venv-sheetsage2/bin/python models/SheetSage2/infer.py source.wav \
  --output cover-score --melody-only
```

`cover-score/score.abc` retains vocal and instrumental melodies while omitting chord symbols. Check the command's exit status and transcription warnings, then listen to the source while reviewing notes, meter, and section order. Transcription errors can carry into the cover.

The same operation is available through Transformers:

```python
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "models/SheetSage2", trust_remote_code=True,
).eval().to("cuda")
result = model.transcribe(
    "source.wav", output_dir="cover-score", melody_only=True,
)
if not result.get("abc") or result.get("abc_error"):
    raise RuntimeError("Transcription did not produce a usable melody score")
print(result.get("warnings", []))
```

For repeatable runs, select reviewed model revisions and retain the downloaded configuration. The model's Python implementation is executed by `trust_remote_code=True`.

## 3. Supply lyrics and a target style

Prepare `cover-request.json` with `style`, `lyrics`, `cot`, and `seed`, following [the original example](../examples/song.json). Set `cot` to `melody`. Use source lyrics you have available, or transcribe and correct the words; align section tags and lyric order with the score. When translating lyrics, match phrasing and syllable counts to the melody.

Switch back to the YuE2 environment:

```bash
.venv/bin/python examples/generate.py --request cover-request.json \
  --abc-file cover-score/score.abc --cot melody --output outputs/cover
```

For a complete score with fixed harmony, use `cot="full"`. Melody-only is recommended for changing styles because it gives the accompaniment more freedom. The [benchmark comparison](benchmarks.md#zero-shot-cover-generation) reports both identity retention and target-style/quality trade-offs.

## Try the included original melody

This example requires only YuE2:

```bash
.venv/bin/python examples/generate.py --request examples/song.json \
  --abc-file examples/melody.abc --cot melody --output outputs/original-melody
```

It tests the score-conditioned interface with original material; it is not a transcription or a benchmark result. The [public cover demos](https://map-yue2.github.io/#cover) illustrate the complete workflow.
