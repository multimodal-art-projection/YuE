# Generate songs with YuE2

Install from the repository root with Python 3.12 and `python -m pip install .`. The supported starting point is a BF16-capable NVIDIA GPU with 24 GB VRAM, one request at a time. The default output is 48 kHz stereo with full symbolic planning and the listening decoder, [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae).

## Style, lyrics, and planning

Put genre, instruments, vocal character, language, and tempo in `style`. Put the words to sing in `lyrics`, with section tags such as `[Verse]` and `[Chorus]`. Start with the [original request](../examples/song.json).

```python
import json
from pathlib import Path
from yue2 import YuE2Pipeline

request = json.loads(Path("examples/song.json").read_text(encoding="utf-8"))
with YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda") as pipe:
    song = pipe(**request)
    song.save_artifacts("outputs/song")
    print(song.truncated)
```

`full` plans melody and chords, `melody` plans only melody, and `off` generates without a symbolic plan. Supplying `abc` uses that composition directly; it requires `full` or `melody`. The native melody input has `Vocal` and `Ins` voices and no chord symbols. Arbitrary ABC dialects may need conversion; the [skill's ABC reference](../skills/yue2-music/references/abc-editing.md) describes the supported notation.

One pipeline call produces one candidate. The benchmark's candidate selection is a separate evaluation step. `cfg_scale` controls text guidance; defaults are ready to use, while changing sampling or guidance may change quality.

The command-line equivalent is:

```bash
yue2 generate --request examples/song.json --output outputs/song-cli
```

## Save and inspect a plan before synthesis

```python
import json
from pathlib import Path
from yue2 import YuE2Pipeline, SymbolicPlan

request = json.loads(Path("examples/song.json").read_text(encoding="utf-8"))
with YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", device="cuda") as pipe:
    plan = pipe.plan(**request)
    plan.save("outputs/plan")
    restored = SymbolicPlan.load("outputs/plan")
    semantic = pipe.generate_semantic(restored)
    latents = pipe.synthesize(semantic)
    audio = pipe.decode(latents)

import soundfile as sf
sf.write("outputs/plan/audio.flac", audio, 48000)
```

`SymbolicPlan.load` restores exact token IDs and checks the saved files. Use it for an unchanged plan. To change the composition, copy the ABC, edit the copy, and submit it as a new `abc` input; do not change a saved plan in place. The end-to-end `save_artifacts()` path is the simplest way to retain a complete generation record.

## Outputs and reproducibility

`save_artifacts()` retains `audio.flac`, `score.abc` when applicable, `plan.json`, semantic tokens, `latent.npy`, effective configuration, timings, model identities, and integrity records. Inspect `result.json` and its `truncated` flags. Saved audio can be playable even when the model hit a token limit. Use a fresh directory for each changed request and retain failures in comparisons.

`from_pretrained` accepts local model directories, `revision`, `vae_revision`, `cache_dir`, and `local_files_only=True`. Pin model and VAE revisions for comparisons. Separate GPUs, runtime versions, or sampling settings can change a seeded generation.

## Listening and evaluation decoders

| Decoder | Use |
|---|---|
| [YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) | Default generation and listening |
| [YuE2-Vae-legacy](https://huggingface.co/m-a-p/YuE2-Vae-legacy) | Reproducing the recorded benchmark protocol |

Choose explicitly with `YuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", vae="m-a-p/YuE2-Vae-legacy")`. Keep listening and evaluation audio in separate directories. When comparing decoders, decode the same cached latents instead of generating a new song:

```bash
python skills/yue2-music/scripts/run_yue2.py decode \
  --source outputs/song --output outputs/song-benchmark \
  --vae m-a-p/YuE2-Vae-legacy
```

The helper verifies the source artifacts and preserves the latent identity. Benchmark claims refer to the stated evaluation model and decoder; a new local generation is not itself a reproduction of the published aggregate.
